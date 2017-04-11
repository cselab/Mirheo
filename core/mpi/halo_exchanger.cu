#include <core/logger.h>
#include <core/cuda_common.h>
#include <core/mpi/halo_exchanger.h>

#include <algorithm>

HaloHelper::HaloHelper(std::string name, const int sizes[3], PinnedBuffer<Particle>* halo)
{
	this->name = name;

	CUDA_Check( cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 0) );

	sendAddrs  .pushStream(stream);
	counts     .pushStream(stream);

	sendAddrs  .resize(27);
	counts     .resize(27);
	recvOffsets.resize(28);

	this->halo = halo;
	this->halo->pushStream(stream);

	auto addrsPtr = sendAddrs.hostPtr();
	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int c = std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
		if (c > 0)
		{
			sendBufs[i].pushStream(stream);
			recvBufs[i].pushStream(stream);

			sendBufs[i].resize( sizes[c-1]*sizeof(Particle) );
			recvBufs[i].resize( sizes[c-1]*sizeof(Particle) );
			addrsPtr[i] = sendBufs[i].devPtr();
		}
	}
	// implicit synchro
	sendAddrs.uploadToDevice();
}

HaloExchanger::HaloExchanger(MPI_Comm& comm, cudaStream_t defStream) :
		nActiveNeighbours(26), defStream(defStream)
{
	MPI_Check( MPI_Comm_dup(comm, &haloComm) );

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
	MPI_Check( MPI_Comm_rank(haloComm, &myrank));

	MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_BYTE, &mpiPartType) );
	MPI_Check( MPI_Type_commit(&mpiPartType) );

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

void HaloExchanger::init()
{
	// Determine halos
	for (int i=0; i<helpers.size(); i++)
		_prepareHalos(i);

	CUDA_Check( cudaStreamSynchronize(defStream) );
}

void HaloExchanger::finalize()
{
	// Send, receive, upload to the device and sync
	for (auto helper : helpers)
		exchange(helper, sizeof(Particle));

	for (auto helper : helpers)
		uploadHalos(helper);

	for (auto helper : helpers)
		CUDA_Check( cudaStreamSynchronize(helper->stream) );
}

void HaloExchanger::exchange(HaloHelper* helper, int typeSize)
{
	auto tagByName = [] (std::string name) {
		static std::hash<std::string> nameHash;
		return (int)( nameHash(name) % 414243 );
	};

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

			MPI_Check( MPI_Irecv(helper->recvBufs[i].hostPtr(), helper->recvBufs[i].size(), MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
			helper->requests.push_back(req);
		}

	// Prepare message sizes and send
	helper->counts.downloadFromDevice();
	auto cntPtr = helper->counts.hostPtr();
	for (int i=0; i<27; i++)
		if (i != 13)
		{
			if (cntPtr[i]*typeSize > helper->sendBufs[i].size())
				die("Preallocated halo buffer %d for %s too small: size %d bytes, but need %d bytes",
						i, pvName.c_str(), helper->sendBufs[i].size(), cntPtr[i]*typeSize);

			if (cntPtr[i] > 0)
				CUDA_Check( cudaMemcpyAsync(helper->sendBufs[i].hostPtr(), helper->sendBufs[i].devPtr(),
						cntPtr[i]*typeSize, cudaMemcpyDeviceToHost, helper->stream) );
		}
	CUDA_Check( cudaStreamSynchronize(helper->stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug3("Sending %s halo to rank %d in dircode %d [%2d %2d %2d], %d particles", pvName.c_str(), dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, cntPtr[i]);
			const int tag = 27 * tagByName(pvName) + i;
			MPI_Check( MPI_Isend(helper->sendBufs[i].hostPtr(), cntPtr[i]*typeSize, MPI_BYTE, dir2rank[i], tag, haloComm, &req) );
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
		msize /= typeSize;
		totalRecvd += msize;

		debug3("Receiving %s halo from rank %d, %d particles", pvName.c_str(), dir2rank[compactedDirs[i]], msize);
	}

	// Fill the holes in the offsets
	helper->recvOffsets[27] = totalRecvd;
	for (int i=0; i<27; i++)
		helper->recvOffsets[i] = std::min(helper->recvOffsets[i+1], helper->recvOffsets[i]);
}


void HaloExchanger::uploadHalos(HaloHelper* helper)
{
	helper->halo->resize(helper->recvOffsets[27], resizeAnew);

	for (int i=0; i < 27; i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(helper->halo->devPtr() + helper->recvOffsets[i], helper->recvBufs[i].hostPtr(),
					msize*sizeof(Particle), cudaMemcpyHostToDevice, helper->stream) );
	}
}


