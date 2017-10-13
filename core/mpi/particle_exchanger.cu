#include "particle_exchanger.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

#include <algorithm>

ExchangeHelper::ExchangeHelper(std::string name, const int datumSize) :
	name(name), datumSize(datumSize)
{
	recvBufSizes.resize(nBuffers);
	recvOffsets. resize(nBuffers+1);
	recvBufs.    resize(nBuffers);
	recvAddrs.   resize_anew(nBuffers);

	sendBufSizes.resize_anew(nBuffers);
	sendBufs.    resize(nBuffers);
	sendAddrs.   resize_anew(nBuffers);

	resizeSendBufs();
	resizeRecvBufs();
}

void ExchangeHelper::resizeSendBufs()
{
	bool changed = false;
	for (int i=0; i<sendBufSizes.size(); i++)
	{
		sendBufs[i].resize_anew(sendBufSizes[i]*datumSize);
		if (sendAddrs[i] != sendBufs[i].devPtr())
			changed = true;

		sendAddrs[i] = sendBufs[i].devPtr();
	}

	if (changed)
	{
		sendAddrs.uploadToDevice(0);
		CUDA_Check( cudaStreamSynchronize(0) );
	}
}

void ExchangeHelper::resizeRecvBufs()
{
	bool changed = false;
	for (int i=0; i<recvBufSizes.size(); i++)
	{
		recvBufs[i].resize_anew(recvBufSizes[i]*datumSize);
		if (recvAddrs[i] != recvBufs[i].devPtr())
			changed = true;

		recvAddrs[i] = recvBufs[i].devPtr();
	}

	if (changed)
	{
		recvAddrs.uploadToDevice(0);
		CUDA_Check( cudaStreamSynchronize(0) );
	}
}

void ExchangeHelper::setDatumSize(int size)
{
	if (size == datumSize) return;

	datumSize = size;
	resizeSendBufs();
	resizeRecvBufs();
}

ParticleExchanger::ParticleExchanger(MPI_Comm& comm) :
		nActiveNeighbours(26)
{
	MPI_Check( MPI_Comm_dup(comm, &haloComm) );

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
	MPI_Check( MPI_Comm_rank(haloComm, &myrank));

	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int coordsNeigh[3];
		for(int c = 0; c < 3; ++c)
			coordsNeigh[c] = coords[c] + d[c];

		MPI_Check( MPI_Cart_rank(haloComm, coordsNeigh, dir2rank + i) );

		dir2sendTag[i] = i;

		int cx = -( i%3 - 1 ) + 1;
		int cy = -( (i/3)%3 - 1 ) + 1;
		int cz = -( i/9 - 1 ) + 1;
		dir2recvTag[i] = (cz*3 + cy)*3 + cx;
	}
}

void ParticleExchanger::init(cudaStream_t stream)
{
	// Derived class determines what to send
	for (int i=0; i<helpers.size(); i++)
		if (needExchange(i)) prepareData(i, stream);
}

void ParticleExchanger::finalize(cudaStream_t stream)
{
	// Internal functions to exchange data
	for (int i=0; i<helpers.size(); i++)
		if (needExchange(i)) send(helpers[i], stream);

	for (int i=0; i<helpers.size(); i++)
		if (needExchange(i)) recv(helpers[i], stream);


	// Derived class unpack implementation
	for (int i=0; i<helpers.size(); i++)
		if (needExchange(i)) combineAndUploadData(i, stream);
}


int ParticleExchanger::tagByName(std::string name)
{
	// TODO: better tagging policy (unique id?)
	static std::hash<std::string> nameHash;
	return (int)( nameHash(name) % (32767 / 27) );
}

void ParticleExchanger::recv(ExchangeHelper* helper, cudaStream_t stream)
{
	std::string pvName = helper->name;

	// Receive sizes
	helper->requests.clear();
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;
			const int tag = 27 * tagByName(pvName) + dir2recvTag[i];

			MPI_Check( MPI_Irecv(helper->recvBufSizes.data() + i, 1, MPI_INT, dir2rank[i], tag, haloComm, &req) );
			helper->requests.push_back(req);
		}

	const int nMessages = helper->requests.size(); // 26 for now
	MPI_Check( MPI_Waitall(nMessages, helper->requests.data(), MPI_STATUSES_IGNORE) );

	// Now do the actual data receive
	int totalRecvd = 0;
	std::fill(helper->recvOffsets.begin(), helper->recvOffsets.end(), std::numeric_limits<int>::max());
	helper->resizeRecvBufs();

	helper->requests.clear();
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;
			const int tag = 27 * tagByName(pvName) + dir2recvTag[i];

			debug3("Receiving %s entities from rank %d, %d entities (buffer %d)",
					pvName.c_str(), dir2rank[i], helper->recvBufSizes[i], i);

			helper->recvOffsets[i] = totalRecvd;
			totalRecvd += helper->recvBufSizes[i];

			MPI_Check( MPI_Irecv(helper->recvBufs[i].hostPtr(), helper->recvBufSizes[i] * helper->datumSize,
					MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			helper->requests.push_back(req);
		}

	// Fill the holes in the offsets
	helper->recvOffsets[27] = totalRecvd;
	for (int i=0; i<27; i++)
		helper->recvOffsets[i] = std::min(helper->recvOffsets[i+1], helper->recvOffsets[i]);

	// Wait for completion
	MPI_Check( MPI_Waitall(nMessages, helper->requests.data(), MPI_STATUSES_IGNORE) );

	debug("Received total %d %s entities", totalRecvd, pvName.c_str());
}

void ParticleExchanger::send(ExchangeHelper* helper, cudaStream_t stream)
{
	std::string pvName = helper->name;

	// Prepare message sizes and send
	helper->sendBufSizes.downloadFromDevice(stream, true);
	auto cntPtr = helper->sendBufSizes.hostPtr();
	for (int i=0; i<27; i++)
		if (i != 13)
		{
			if (cntPtr[i] * helper->datumSize > helper->sendBufs[i].size())
				die("Allocated buffer %d for %s too small: size %d bytes, but need %d bytes",
						i, pvName.c_str(), helper->sendBufs[i].size(), cntPtr[i] * helper->datumSize);

			if (cntPtr[i] > 0)
				CUDA_Check( cudaMemcpyAsync(helper->sendBufs[i].hostPtr(), helper->sendBufs[i].devPtr(),
						cntPtr[i] * helper->datumSize, cudaMemcpyDeviceToHost, stream) );
		}
	CUDA_Check( cudaStreamSynchronize(stream) );

	MPI_Request req;
	int totSent = 0;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug3("Sending %s entities to rank %d in dircode %d [%2d %2d %2d], %d entities",
					pvName.c_str(), dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, cntPtr[i]);
			const int tag = 27 * tagByName(pvName) + dir2sendTag[i];

			MPI_Check( MPI_Isend(cntPtr+i, 1, MPI_INT, dir2rank[i], tag, haloComm, &req) );
			MPI_Check( MPI_Request_free(&req) );
			MPI_Check( MPI_Isend(helper->sendBufs[i].hostPtr(), cntPtr[i] * helper->datumSize, MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			MPI_Check( MPI_Request_free(&req) );

			totSent += cntPtr[i];
		}
	debug("Sent total %d %s entities", totSent, pvName.c_str());
}


