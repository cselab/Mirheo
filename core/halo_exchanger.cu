#include "datatypes.h"
#include "containers.h"
#include "halo_exchanger.h"
#include "celllist.h"
#include "logger.h"

#include <vector>
#include <thread>
#include <algorithm>
#include <unistd.h>

__global__ void getHalos(const int* __restrict__ cellsStart, const float4* __restrict__ xyzouvwo, const int3 ncells, const int totcells, const float3 length,
		const int64_t __restrict__ dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int variant = blockIdx.y;
	const int tid = threadIdx.x;
	int cid;
	int cx, cy, cz;

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

	if (__all(!valid) && tid > 27) return;

	int2 start_size = valid ? decodeStartSize(cellsStart[cid]) : make_int2(0, 0);

	// Use shared memory to decrease global atomics
	// We're sending to max 7 halos (corner)
	int validHalos[7];
	int haloOffset[7] = {};
	int current = 0;

	// Total number of elements written to halos by this block
	__shared__ int blockSum[27];
	if (tid < 27) blockSum[tid] = 0;

	__syncthreads();

	for (int ix = min(cx, 1); ix <= max(cx, 1); ix++)
		for (int iy = min(cy, 1); iy <= max(cy, 1); iy++)
			for (int iz = min(cz, 1); iz <= max(cz, 1); iz++)
			{
				if (ix == 1 && iy == 1 && iz == 1) continue;

				const int bufId = (iz*3 + iy)*3 + ix;
				validHalos[current] = bufId;
				haloOffset[current] = atomicAdd(blockSum + bufId, start_size.y);
				current++;
			}

	__syncthreads();

	if (tid < 27 && blockSum[tid] > 0)
		blockSum[tid] = atomicAdd(counts + tid, blockSum[tid]);

	__syncthreads();

#pragma unroll 3
	for (int i=0; i<current; i++)
	{
		const int bufId = validHalos[i];
		const int myid  = blockSum[bufId] + haloOffset[i];

		const int ix = bufId % 3;
		const int iy = (bufId / 3) % 3;
		const int iz = bufId / 9;
		const float4 shift{ length.x*(ix-1), length.y*(iy-1), length.z*(iz-1), 0 };

		for (int i = 0; i < start_size.y; i++)
		{
			const int dstInd = 2*(myid         + i);
			const int srcInd = 2*(start_size.x + i);

			float4 tmp1 = xyzouvwo[srcInd] - shift;
			float4 tmp2 = xyzouvwo[srcInd+1];

			float4* addr = (float4*)dests[bufId];
			addr[dstInd + 0] = tmp1;
			addr[dstInd + 1] = tmp2;
		}
	}
}

HaloExchanger::HaloExchanger(MPI_Comm& comm) : nActiveNeighbours(26)
{
	MPI_Check( MPI_Comm_dup(comm, &haloComm));

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

void HaloExchanger::attach(ParticleVector* pv, int ndens)
{
	particleVectors.push_back(pv);

	helpers.resize(helpers.size() + 1);
	HaloHelper& helper = helpers[helpers.size() - 1];

	helper.sendAddrs.resize(27);
	helper.counts.resize(27);

	CUDA_Check( cudaStreamCreateWithPriority(&helper.stream, cudaStreamNonBlocking, -10) );

	const int maxdim = std::max({pv->ncells.x, pv->ncells.y, pv->ncells.z});

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

void HaloExchanger::exchange()
{
	for (int i=0; i<particleVectors.size(); i++)
		_initialize(i);

	for (int i=0; i<particleVectors.size(); i++)
	{
		CUDA_Check( cudaStreamSynchronize(helpers[i].stream) );
		send(i);
	}

	for (int i=0; i<particleVectors.size(); i++)
		receive(i);
}

void HaloExchanger::_initialize(int n)
{
	auto pv = particleVectors[n];
	HaloHelper& helper = helpers[n];

	helper.requests.clear();
	for (int i=0; i<nActiveNeighbours; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			MPI_Request req;
			const int tag = 27*n + i;
			MPI_Check( MPI_Irecv(helper.recvBufs[i].hostdata, helper.recvBufs[i].size, mpiPartType, dir2rank[i], tag, haloComm, &req) );
			helper.requests.push_back(req);
		}


	const int maxdim = std::max({pv->ncells.x, pv->ncells.y, pv->ncells.z});
	const int nthreads = 32;
	helper.counts.clear(helper.stream);

	debug("Preparing %d-th halo on the device", n);
	getHalos<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, helper.stream >>>
			(pv->cellsStart.devdata, (float4*)pv->coosvels.devdata, pv->ncells, pv->totcells, pv->length,
			 (int64_t*)helper.sendAddrs.devdata, helper.counts.devdata);
}

void HaloExchanger::send(int n)
{
	auto pv = particleVectors[n];
	HaloHelper& helper = helpers[n];

	helper.counts.synchronize(synchronizeHost, helper.stream);
	debug("Downloading %d-th halo", n);

	// Can't use synchronize here because we actually have only helper.counts[i] elements
	for (int i=0; i<27; i++)
		if (i != 13)
			CUDA_Check( cudaMemcpyAsync(helper.sendBufs[i].hostdata, helper.sendBufs[i].devdata,
					helper.counts[i]*sizeof(Particle), cudaMemcpyDeviceToHost, helper.stream) );
	CUDA_Check( cudaStreamSynchronize(helper.stream) );

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] >= 0)
		{
			debug("Sending %d-th halo to rank %d in dircode %d [%2d %2d %2d], size %d", n, dir2rank[i], i, i%3 - 1, (i/3)%3 - 1, i/9 - 1, helper.counts[i]);
			const int tag = 27*n + i;
			MPI_Check( MPI_Isend(helper.sendBufs[i].hostdata, helper.counts[i], mpiPartType, dir2rank[i], tag, haloComm, &req) );
		}
}

void HaloExchanger::receive(int n)
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

	pv->halo.resize(totalRecvd, resizeAnew, helper.stream);

	// Load onto the device
	for (int i=0; i<nMessages; i++)
	{
		debug("Receiving %d-th halo from rank %d, size %d",
				n, dir2rank[compactedDirs[i]], compactedDirs[i], compactedDirs[i]%3 - 1, (compactedDirs[i]/3)%3 - 1, compactedDirs[i]/9 - 1, offsets[i+1] - offsets[i]);
		CUDA_Check( cudaMemcpyAsync(pv->halo.devdata + offsets[i], helper.recvBufs[compactedDirs[i]].hostdata,
				(offsets[i+1] - offsets[i])*sizeof(Particle), cudaMemcpyHostToDevice, helper.stream) );
	}

	CUDA_Check( cudaStreamSynchronize(helper.stream) );
}
