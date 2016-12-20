#include "datatypes.h"
#include "containers.h"
#include "redistributor.h"
#include "celllist.h"
#include "logger.h"

#include <vector>
#include <thread>
#include <algorithm>
#include <unistd.h>

__global__ void getExitingParticles(const int* __restrict__ cellsStart, float4* xyzouvwo,
		const int3 ncells, const int totcells, const float3 length, const float3 domainStart,
		const int64_t __restrict__ dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int variant = blockIdx.y;
	int cid;
	int cx, cy, cz;

	// Select all the boundary cells WITHOUT repetitions
	bool valid = true;

	if (variant <= 1)  // x
	{
		if (gid >= ncells.y*ncells.z) valid = false;
		cx = variant * (ncells.x - 1);
		cy = gid % ncells.y;
		cz = gid / ncells.y;
		cid = encode(cx, cy, cz, ncells);
	}
	else if (variant <= 3)  // y
	{
		if (gid >= ncells.x*ncells.z) valid = false;
		cx = gid % ncells.x;
		cy = (variant - 2) * (ncells.y - 1);
		cz = gid / ncells.x;
		cid = encode(cx, cy, cz, ncells);
	}
	else   // z
	{
		if (gid >= ncells.x*ncells.y) valid = false;
		cx = gid % ncells.x;
		cy = gid / ncells.x;
		cz = (variant - 4) * (ncells.z - 1);
		cid = encode(cx, cy, cz, ncells);
	}

	valid &= cid < totcells;

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

	int2 start_size = valid ? decodeStartSize(cellsStart[cid]) : make_int2(0, 0);

	for (int i = 0; i < start_size.y; i++)
	{
		const int srcId = start_size.x + i;
		const float4 coo = xyzouvwo[2*srcId];
		const float4 vel = xyzouvwo[2*srcId+1];

		int px = getCellIdAlongAxis<false>(coo.x, domainStart.x, ncells.x, 1.0f);
		int py = getCellIdAlongAxis<false>(coo.y, domainStart.y, ncells.y, 1.0f);
		int pz = getCellIdAlongAxis<false>(coo.z, domainStart.z, ncells.z, 1.0f);

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
			const float4 shift{ length.x*(px-1), length.y*(py-1), length.z*(pz-1), 0 };

			int myid = atomicAdd(counts + bufId, 1);

			const int dstInd = 2*myid;

			float4* addr = (float4*)dests[bufId];
			addr[dstInd + 0] = coo - shift;
			addr[dstInd + 1] = vel;

			// mark the particle as exited to assist cell-list building
			xyzouvwo[2*srcId] = make_float4(-1000);
		}
	}
}

Redistributor::Redistributor(MPI_Comm& comm, IniParser& config) : nActiveNeighbours(26), config(config)
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

void Redistributor::attach(ParticleVector* pv, int ndens)
{
	particleVectors.push_back(pv);

	helpers.resize(helpers.size() + 1);
	auto& helper = helpers[helpers.size() - 1];

	helper.sendAddrs.resize(27);
	helper.counts.resize(27);

	const int maxdim = std::max({pv->ncells.x, pv->ncells.y, pv->ncells.z});

	CUDA_Check( cudaStreamCreateWithPriority(&helper.stream, cudaStreamNonBlocking, -10) );

	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3 - 1, (i/3) % 3 - 1, i/9 - 1 };

		int c = std::abs(d[0]) + std::abs(d[1]) + std::abs(d[2]);
		if (c > 0)
		{
			helper.sendBufs[i].resize( 3 * ndens * pow(maxdim, 3 - c) );
			helper.recvBufs[i].resize( 3 * ndens * pow(maxdim, 3 - c) );
			helper.sendAddrs[i] = (float4*)helper.sendBufs[i].devdata;
		}
	}
	helper.sendAddrs.synchronize(synchronizeDevice);
}

void Redistributor::redistribute(float dt)
{
	for (int i=0; i<particleVectors.size(); i++)
		_initialize(i, dt);

	for (int i=0; i<particleVectors.size(); i++)
	{
		CUDA_Check( cudaStreamSynchronize(helpers[i].stream) );
		send(i);
	}

	for (int i=0; i<particleVectors.size(); i++)
		receive(i);
}

void Redistributor::_initialize(int n, float dt)
{
	auto& pv = *particleVectors[n];
	auto& helper = helpers[n];

	const int maxdim = std::max({pv.ncells.x, pv.ncells.y, pv.ncells.z});
	const int nthreads = 32;
	helper.counts.clear(helper.stream);
	auto config = this->config;

	debug("Preparing %d-th leaving particles on the device", n);

	helper.requests.clear();
	for (int i=0; i<nActiveNeighbours; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;
			const int tag = 27*n + i;
			MPI_Check( MPI_Irecv(helper.recvBufs[i].hostdata, helper.recvBufs[i].size, mpiPartType, dir2rank[i], tag, redComm, &req) );
			helper.requests.push_back(req);
		}

	getExitingParticles<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, helper.stream >>>
				(pv.cellsStart.devdata, (float4*)pv.coosvels.devdata,
				 pv.ncells, pv.totcells, pv.length, pv.domainStart,
				 (int64_t*)helper.sendAddrs.devdata, helper.counts.devdata);
}

void Redistributor::send(int n)
{
	auto& pv = particleVectors[n];
	auto& helper = helpers[n];

	helper.counts.synchronize(synchronizeHost, helper.stream);
	debug("Downloading %d-th leaving particles", n);
	for (int i=0; i<27; i++)
		if (i != 13)
			CUDA_Check( cudaMemcpyAsync(helper.sendBufs[i].hostdata, helper.sendBufs[i].devdata,
					helper.counts[i]*sizeof(Particle), cudaMemcpyDeviceToHost, helper.stream) );
	CUDA_Check( cudaStreamSynchronize(helper.stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug("Sending %d-th redistribution to rank %d in dircode %d [%2d %2d %2d], size %d",
					n, dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, helper.counts[i]);

			const int tag = 27*n + i;
			MPI_Check( MPI_Isend(helper.sendBufs[i].hostdata, helper.counts[i], mpiPartType, dir2rank[i], tag, redComm, &req) );
		}
}

void Redistributor::receive(int n)
{
	auto& helper = helpers[n];
	auto& pv = particleVectors[n];

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
	pv->resize(oldsize + totalRecvd, resizePreserve, helper.stream);
	pv->received = totalRecvd; // TODO: get rid of this

	// Load onto the device
	for (int i=0; i<nMessages; i++)
	{
		debug("Receiving %d-th redistribution from rank %d in dircode %d [%2d %2d %2d], size %d",
				n, dir2rank[compactedDirs[i]], compactedDirs[i], compactedDirs[i]%3 - 1, (compactedDirs[i]/3)%3 - 1, compactedDirs[i]/9 - 1, offsets[i+1] - offsets[i]);
		CUDA_Check( cudaMemcpyAsync(pv->coosvels.devdata + oldsize + offsets[i], helper.recvBufs[compactedDirs[i]].hostdata,
				(offsets[i+1] - offsets[i])*sizeof(Particle), cudaMemcpyHostToDevice, helper.stream) );
	}

	CUDA_Check( cudaStreamSynchronize(helper.stream) );
}


