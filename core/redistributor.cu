#include <core/celllist.h>
#include <core/containers.h>
#include <core/redistributor.h>
#include <core/helper_math.h>

#include <vector>
#include <thread>
#include <algorithm>

__global__ void getExitingParticles(float4* xyzouvwo,
		CellListInfo cinfo, const int* __restrict__ cellsStart,
		float4* __restrict__ dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int variant = blockIdx.y;
	int cid;
	int cx, cy, cz;

	const int3 ncells = cinfo.ncells;

	// Select all the boundary cells WITHOUT repetitions
	bool valid = true;

	if (variant <= 1)  // x
	{
		if (gid >= ncells.y * ncells.z) valid = false;
		cx = variant * (ncells.x - 1);
		cy = gid % ncells.y;
		cz = gid / ncells.y;
		cid = cinfo.encode(cx, cy, cz);
	}
	else if (variant <= 3)  // y
	{
		if (gid >= ncells.x * ncells.z) valid = false;
		cx = gid % ncells.x;
		cy = (variant - 2) * (ncells.y - 1);
		cz = gid / ncells.x;
		cid = cinfo.encode(cx, cy, cz);
	}
	else   // z
	{
		if (gid >= ncells.x * ncells.y) valid = false;
		cx = gid % ncells.x;
		cy = gid / ncells.x;
		cz = (variant - 4) * (ncells.z - 1);
		cid = cinfo.encode(cx, cy, cz);
	}

	valid &= cid < cinfo.totcells;

	// Find side codes
	if (cx == 0) cx = 0;
	else if (cx == ncells.x-1) cx = 2;
	else cx = 1;

	if (cy == 0) cy = 0;
	else if (cy == ncells.y-1) cy = 2;
	else cy = 1;

	if (cz == 0) cz = 0;
	else if (cz == ncells.z-1) cz = 2;
	else cz = 1;

	// Exclude cells already covered by other variants
	if ( (variant == 0 || variant == 1) && (cz != 1 || cy != 1) ) valid = false;
	if ( (variant == 2 || variant == 3) && (cz != 1) ) valid = false;

	if (!valid) return;

	// The following is called for every outer cell and exactly once for each
	//
	// Now for each cell we check its every particle if it needs to move

	int2 start_size = valid ? cinfo.decodeStartSize(cellsStart[cid]) : make_int2(0, 0);

#pragma unroll 2
	for (int i = 0; i < start_size.y; i++)
	{
		const int srcId = start_size.x + i;
		const float4 coo = xyzouvwo[2*srcId];
		const float4 vel = xyzouvwo[2*srcId+1];

		int px = cinfo.getCellIdAlongAxis<0, false>(coo.x);
		int py = cinfo.getCellIdAlongAxis<1, false>(coo.y);
		int pz = cinfo.getCellIdAlongAxis<2, false>(coo.z);

		if (px < 0) px = 0;
		else if (px >= ncells.x) px = 2;
		else px = 1;

		if (py < 0) py = 0;
		else if (py >= ncells.y) py = 2;
		else py = 1;

		if (pz < 0) pz = 0;
		else if (pz >= ncells.z) pz = 2;
		else pz = 1;

		if (px*py*pz != 1) // this means that the particle has to leave
		{
			const int bufId = (pz*3 + py)*3 + px;
			const float4 shift{ cinfo.length.x*(px-1), cinfo.length.y*(py-1), cinfo.length.z*(pz-1), 0 };

			int myid = atomicAdd(counts + bufId, 1);

			const int dstInd = 2*myid;

			float4* addr = dests[bufId];
			float4 newcoo = coo - shift;
			newcoo.w = coo.w;
			addr[dstInd + 0] = newcoo;
			addr[dstInd + 1] = vel;

			// mark the particle as exited to assist cell-list building
			xyzouvwo[2*srcId] = make_float4(-1000.0f, -1000.0f, -1000.0f, coo.w);
		}
	}
}

Redistributor::Redistributor(MPI_Comm& comm) : nActiveNeighbours(26)
{
	MPI_Check( MPI_Comm_dup(comm, &redComm));

	int dims[3], periods[3], coords[3];
	MPI_Check( MPI_Cart_get (redComm, 3, dims, periods, coords) );
	MPI_Check( MPI_Comm_rank(redComm, &myrank));

	MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_BYTE, &mpiPartType) );
	MPI_Check( MPI_Type_commit(&mpiPartType) );

	int active = 0;
	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int coordsNeigh[3];
		for(int c = 0; c < 3; ++c)
			coordsNeigh[c] = coords[c] + d[c];

		MPI_Check( MPI_Cart_rank(redComm, coordsNeigh, dir2rank + i) );
		if (dir2rank[i] >= 0 && i != 13)
			compactedDirs[active++] = i;
	}
}

void Redistributor::attach(ParticleVector* pv, CellList* cl)
{
	const float ndens = (float)pv->np / (cl->ncells.x * cl->ncells.y * cl->ncells.z * cl->rc*cl->rc*cl->rc);
	particlesAndCells.push_back({pv, cl});

	helpers.resize(helpers.size() + 1);
	auto& helper = helpers[helpers.size() - 1];

	helper.sendAddrs.resize(27);
	helper.counts.resize(27);

	CUDA_Check( cudaStreamCreateWithPriority(&helper.stream, cudaStreamNonBlocking, -10) );

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});

	auto addrsPtr = helper.sendAddrs.hostPtr();
	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int c = std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
		if (c > 0)
		{
			helper.sendBufs[i].resize( 3 * ndens * pow(maxdim, 3 - c) + 64);
			helper.recvBufs[i].resize( 3 * ndens * pow(maxdim, 3 - c) + 64);
			addrsPtr[i] = (float4*)helper.sendBufs[i].devPtr();

			helper.sendBufs[i].pushStream(helper.stream);
		}
	}
	helper.sendAddrs.uploadToDevice();

	helper.counts.pushStream(helper.stream);
	helper.sendAddrs.pushStream(helper.stream);
}

void Redistributor::redistribute()
{
	for (int i=0; i<particlesAndCells.size(); i++)
		_initialize(i);

	for (int i=0; i<particlesAndCells.size(); i++)
	{
		CUDA_Check( cudaStreamSynchronize(helpers[i].stream) );
		send(i);
	}

	for (int i=0; i<particlesAndCells.size(); i++)
		receive(i);
}

void Redistributor::_initialize(int n)
{
	auto pv = particlesAndCells[n].first;
	auto cl = particlesAndCells[n].second;
	auto& helper = helpers[n];

	helper.counts.clear();
	helper.requests.clear();
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;

			// Invert the direction index
			const int cx = -( i%3 - 1 ) + 1;
			const int cy = -( (i/3)%3 - 1 ) + 1;
			const int cz = -( i/9 - 1 ) + 1;

			const int dirCode = (cz*3 + cy)*3 + cx;
			const int tag = 27*n + dirCode;

			MPI_Check( MPI_Irecv(helper.recvBufs[i].hostPtr(), helper.recvBufs[i].size(), mpiPartType, dir2rank[i], tag, redComm, &req) );
			helper.requests.push_back(req);
		}

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
	const int nthreads = 32;
	getExitingParticles<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, helper.stream >>>
				( (float4*)pv->coosvels.devPtr(), cl->cellInfo(), cl->cellsStart.devPtr(), helper.sendAddrs.devPtr(), helper.counts.devPtr() );

	helper.counts.downloadFromDevice(false);
}

void Redistributor::send(int n)
{
	auto pv = particlesAndCells[n].first;
	auto& helper = helpers[n];

	// Wait for the previous downloads
	CUDA_Check( cudaStreamSynchronize(helper.stream) );

	int totalLeaving = 0;
	auto cntPtr = helper.counts.hostPtr();
	for (int i=0; i<27; i++)
		if (i != 13)
			totalLeaving += cntPtr[i];

	debug2("Preparing %d leaving %s particles on the device", totalLeaving, pv->name.c_str());


	for (int i=0; i<27; i++)
		if (i != 13)
			CUDA_Check( cudaMemcpyAsync(helper.sendBufs[i].hostPtr(), helper.sendBufs[i].devPtr(),
					cntPtr[i]*sizeof(Particle), cudaMemcpyDeviceToHost, helper.stream) );
	CUDA_Check( cudaStreamSynchronize(helper.stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug3("Sending %s redistribution to rank %d in dircode %d [%2d %2d %2d], size %d",
					pv->name.c_str(), dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, cntPtr[i]);

			const int tag = 27*n + i;
			MPI_Check( MPI_Isend(helper.sendBufs[i].hostPtr(), cntPtr[i], mpiPartType, dir2rank[i], tag, redComm, &req) );
		}
}

void Redistributor::receive(int n)
{
	auto pv = particlesAndCells[n].first;
	auto& helper = helpers[n];

	// Wait until messages are arrived
	const int nMessages = helper.requests.size();
	std::vector<MPI_Status> statuses(nMessages);
	MPI_Check( MPI_Waitall(nMessages, &helper.requests[0], &statuses[0]) );

	// Excl scan of message sizes to know where we will upload them
	std::vector<int> offsets(nMessages+1);
	int totalRecvd = 0;
	for (int i=0; i<nMessages; i++)
	{
		offsets[i] = totalRecvd;

		int msize;
		MPI_Check( MPI_Get_count(&statuses[i], mpiPartType, &msize) );
		totalRecvd += msize;
	}
	offsets[nMessages] = totalRecvd;

	int oldsize = pv->np;
	pv->resize(oldsize + totalRecvd, resizePreserve);
	pv->received = totalRecvd; // TODO: get rid of this
	debug2("Receiving %d total %s particles", totalRecvd, pv->name.c_str());

	// Load onto the device
	for (int i=0; i<nMessages; i++)
	{
		debug3("Receiving %s redistribution from rank %d in dircode %d [%2d %2d %2d], size %d",
				pv->name.c_str(), dir2rank[compactedDirs[i]], compactedDirs[i],
				compactedDirs[i]%3 - 1, (compactedDirs[i]/3)%3 - 1, compactedDirs[i]/9 - 1, offsets[i+1] - offsets[i]);
		CUDA_Check( cudaMemcpyAsync(pv->coosvels.devPtr() + oldsize + offsets[i], helper.recvBufs[compactedDirs[i]].hostPtr(),
				(offsets[i+1] - offsets[i])*sizeof(Particle), cudaMemcpyHostToDevice, helper.stream) );
	}

	CUDA_Check( cudaStreamSynchronize(helper.stream) );
}


