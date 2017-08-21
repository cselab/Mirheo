#include <core/logger.h>
#include <core/cuda_common.h>
#include <core/mpi/particle_exchanger.h>

#include <algorithm>

ExchangeHelper::ExchangeHelper(std::string name, const int datumSize, const int sizes[3])
{
	this->name = name;
	this->datumSize = datumSize;

	CUDA_Check( cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0) );

	sendAddrs  .resize(27, stream);
	bufSizes   .resize(27, stream);
	recvOffsets.resize(28);

	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int c = std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
		if (c > 0)
		{
			sendBufs[i].resize( sizes[c-1]*datumSize, stream );
			recvBufs[i].resize( sizes[c-1]*datumSize, stream );
			sendAddrs[i] = sendBufs[i].devPtr();
		}
	}
	// implicit synchro
	sendAddrs.uploadToDevice(stream);

	CUDA_Check( cudaStreamSynchronize(stream) );
}

ParticleExchanger::ParticleExchanger(MPI_Comm& comm) :
		nActiveNeighbours(26)
{
	MPI_Check( MPI_Comm_dup(comm, &haloComm) );

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
	MPI_Check( MPI_Comm_rank(haloComm, &myrank));

	int active = 0;
	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int coordsNeigh[3];
		for(int c = 0; c < 3; ++c)
			coordsNeigh[c] = coords[c] + d[c];

		MPI_Check( MPI_Cart_rank(haloComm, coordsNeigh, dir2rank + i) );
		if (dir2rank[i] >= 0 && i != 13)
			compactedDirs[active++] = i;
	}
}

void ParticleExchanger::init(cudaStream_t defStream)
{
	// Post recv
	for (auto helper : helpers)
		postRecv(helper);

	// Determine halos
	for (int i=0; i<helpers.size(); i++)
		prepareData(i, defStream);

	CUDA_Check( cudaStreamSynchronize(defStream) );
}

void ParticleExchanger::finalize()
{
	// Send, receive, upload to the device and sync
	for (auto helper : helpers)
		sendWait(helper);

	for (int i=0; i<helpers.size(); i++)
		combineAndUploadData(i);

	for (auto helper : helpers)
		CUDA_Check( cudaStreamSynchronize(helper->stream) );
}

inline int tagByName(std::string name)
{
	static std::hash<std::string> nameHash;
	return (int)( nameHash(name) % 414243 );
}

void ParticleExchanger::postRecv(ExchangeHelper* helper)
{
	std::string pvName = helper->name;

	// Post receives
	helper->requests.clear();
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;

			// Invert the direction index
			const int cx = -( i%3 - 1 ) + 1;
			const int cy = -( (i/3)%3 - 1 ) + 1;
			const int cz = -( i/9 - 1 ) + 1;

			const int invDirCode = (cz*3 + cy)*3 + cx;
			const int tag = 27 * tagByName(pvName) + invDirCode;

			MPI_Check( MPI_Irecv(helper->recvBufs[i].hostPtr(), helper->recvBufs[i].size() * helper->datumSize, MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			helper->requests.push_back(req);
		}
}

void ParticleExchanger::sendWait(ExchangeHelper* helper)
{
	std::string pvName = helper->name;

	// Prepare message sizes and send
	helper->bufSizes.downloadFromDevice(helper->stream, true);
	auto cntPtr = helper->bufSizes.hostPtr();
	for (int i=0; i<27; i++)
		if (i != 13)
		{
			if (cntPtr[i] * helper->datumSize > helper->sendBufs[i].size())
				die("Preallocated halo buffer %d for %s too small: size %d bytes, but need %d bytes",
						i, pvName.c_str(), helper->sendBufs[i].size(), cntPtr[i] * helper->datumSize);

			if (cntPtr[i] > 0)
				CUDA_Check( cudaMemcpyAsync(helper->sendBufs[i].hostPtr(), helper->sendBufs[i].devPtr(),
						cntPtr[i] * helper->datumSize, cudaMemcpyDeviceToHost, helper->stream) );
		}
	CUDA_Check( cudaStreamSynchronize(helper->stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug3("Sending %s halo to rank %d in dircode %d [%2d %2d %2d], %d entities", pvName.c_str(), dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, cntPtr[i]);
			const int tag = 27 * tagByName(pvName) + i;
			MPI_Check( MPI_Isend(helper->sendBufs[i].hostPtr(), cntPtr[i] * helper->datumSize, MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			MPI_Check( MPI_Request_free(&req) );
		}

	// Wait until messages are arrived
	const int nMessages = helper->requests.size();
	std::vector<MPI_Status> statuses(nMessages);
	MPI_Check( MPI_Waitall(nMessages, helper->requests.data(), statuses.data()) );

	// Excl scan of message sizes to know where we will upload them
	int totalRecvd = 0;
	std::fill(helper->recvOffsets.begin(), helper->recvOffsets.end(), std::numeric_limits<int>::max());
	for (int i=0; i<nMessages; i++)
	{
		helper->recvOffsets[compactedDirs[i]] = totalRecvd;

		int msize;
		MPI_Check( MPI_Get_count(&statuses[i], MPI_BYTE, &msize) );
		totalRecvd += msize / helper->datumSize;

		debug3("Receiving %s halo from rank %d, %d entities", pvName.c_str(), dir2rank[compactedDirs[i]], msize /  helper->datumSize);
	}

	// Fill the holes in the offsets
	helper->recvOffsets[27] = totalRecvd;
	for (int i=0; i<27; i++)
		helper->recvOffsets[i] = std::min(helper->recvOffsets[i+1], helper->recvOffsets[i]);
}


