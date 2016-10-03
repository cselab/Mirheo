#include "datatypes.h"
#include "containers.h"
#include "halo_exchanger.h"

#include <vector>
#include <thread>
#include <algorithm>
#include <unistd.h>

__global__ void getHalos(const int* __restrict__ cellsStart, const float4* __restrict__ xyzouvwo, const int3 ncells, const int totcells,
		float4* dst0, float4* dst1, float4* dst2, float4* dst3, int count[4], int* limits)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int variant = blockIdx.y;
	int cid;

	if (variant <= 1)
		cid = gid * ncells.x  +  (ncells.x-1) * variant; // x = 0,   x = nx - 1
	else
		cid = (gid % ncells.x) + (gid / ncells.x) * ncells.x*(ncells.y-1)  +  ncells.x * (ncells.y-1) * (variant - 2);  // y = 0,   y = ny - 1

	if (cid >= totcells) return;

	volatile __shared__ int shIndex;
	__shared__ int bsize;
	bsize = 0;

	const int pstart = cellsStart[cid];
	const int pend   = cellsStart[cid+1];

	int myid = atomicAdd(&bsize, pend-pstart);

	__syncthreads();

	if (threadIdx.x == 0)
		shIndex = atomicAdd(count + variant, bsize);

	__syncthreads();

	myid += shIndex;

	float4* dest[4] = {dst0, dst1, dst2, dst3};

	for (int i = 0; i < pend-pstart; i++)
	{
		const int dstInd = 2*(myid   + i);
		const int srcInd = 2*(pstart + i);

		float4 tmp1 = xyzouvwo[srcInd];
		float4 tmp2 = xyzouvwo[srcInd];
		tmp1.w = __int_as_float(pstart + i);
		tmp2.w = __int_as_float(cid);

		dest[variant][dstInd + 0] = tmp1;
		dest[variant][dstInd + 1] = tmp2;
	}

	if (gid == 0 && variant == 0)
	{
		limits[0] = cellsStart[ncells.x * ncells.y];
		limits[1] = cellsStart[totcells - ncells.x * ncells.y];
	}
}

HaloExchanger::HaloExchanger(MPI_Comm& comm) : nActiveNeighbours(26)
{
	logger.MPI_Check( MPI_Comm_dup(comm, &haloComm));

	int dims[3], periods[3], coords[3];
	logger.MPI_Check( MPI_Cart_get (haloComm, 3, dims, periods, coords) );
	logger.MPI_Check( MPI_Comm_rank(haloComm, &myrank));

	logger.MPI_Check( MPI_Type_contiguous(sizeof(Particle), MPI_BYTE, &mpiParticleType) );

	for(int i = 0; i < 27; ++i)
	{
		int d[3] = { i%3, (i/3) % 3, i/9 };

		int coordsNeigh[3];
		for(int c = 0; c < 3; ++c)
			coordsNeigh[c] = coords[c] + d[c];

		logger.MPI_Check( MPI_Cart_rank(haloComm, coordsNeigh, dir2rank + i) );
	}
}

void HaloExchanger::attach(ParticleVector* pv)
{
	particleVectors.push_back(pv);

	helpers.resize(helpers.size() + 1);
	HaloHelper& helper = helpers[helpers.size() - 1];

	helper.counts.resize(4);
	helper.limits.resize(2);

	logger.CUDA_Check( cudaStreamCreate(&helper.stream) );
}

void HaloExchanger::exchangeInit()
{
	for (int i=0; i<particleVectors.size(); i++)
	{
		helpers[i].thread = std::thread([&] {
			send(i);
			receive(i);
		});
	}
}

void HaloExchanger::exchangeFinalize()
{
	for (auto& h : helpers)
		h.thread.join();
}

void HaloExchanger::send(int n)
{
	auto pv = particleVectors[n];
	HaloHelper& helper = helpers[n];

	const int maxdim = std::max({pv->ncells.x, pv->ncells.y, pv->ncells.z});
	const int nthreads = 128;

	helper.counts.clear(helper.stream);
	// TODO: need number density here
	for (int i=0; i<6; i++)
		helper.sides[i].resize(maxdim*maxdim * 10 * 4);

	printf("md = %d\n", maxdim);

	getHalos<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 4, 1),  dim3(nthreads, 1, 1), 0, helper.stream >>>
			(pv->cellsStart.devdata, (float4*)pv->coosvels.devdata, pv->ncells, pv->totcells,
					(float4*)helper.sides[0].devdata,
					(float4*)helper.sides[1].devdata,
					(float4*)helper.sides[2].devdata,
					(float4*)helper.sides[3].devdata, helper.counts.devdata, helper.limits.devdata);

	// This can be easily removed by keeping stuff in gpu mem
	helper.limits.synchronize(synchronizeHost, helper.stream);
	helper.counts.synchronize(synchronizeHost, helper.stream);

	logger.CUDA_Check( cudaMemcpyAsync(helper.sides[4].devdata, pv->coosvels.devdata,
			helper.limits[0] * sizeof(Particle),cudaMemcpyDeviceToHost, helper.stream) );

	logger.CUDA_Check( cudaMemcpyAsync(helper.sides[5].devdata, pv->coosvels.devdata + helper.limits[1],
			(pv->np - helper.limits[1]) * sizeof(Particle), cudaMemcpyDeviceToHost, helper.stream) );

	logger.CUDA_Check( cudaStreamSynchronize(helper.stream) );

	for (auto& buf : helper.sendBufs)
		buf.resize(0);

	auto processOneSide = [&] (Particle* srcs, int n)
	{
		for (int pid=0; pid<n; pid++)
		{
			auto& p = srcs[pid];
			const int cid = p.i2;

			int cx, cy, cz;
			cx = cid % pv->ncells.x;
			cy = (cid / pv->ncells.x) % pv->ncells.y;
			cz = cid / (pv->ncells.x * pv->ncells.y);

			if (cx == 0) cx = 0;
			else if (cx == pv->ncells.x-1) cx = 2;
			else cx = 1;

			if (cy == 0) cy = 0;
			else if (cy == pv->ncells.y-1) cy = 2;
			else cy = 1;

			if (cz == 0) cz = 0;
			else if (cz == pv->ncells.z-1) cz = 2;
			else cz = 1;

			for (int ix = min(cx, 1); ix <= max(cx, 1); ix++)
				for (int iy = min(cy, 1); iy <= max(cy, 1); iy++)
					for (int iz = min(cz, 1); iz <= max(cz, 1); iz++)
					{
						if (cx == 1 && cy == 1 && cz == 1) continue;

						int bufId = (cz*3 + cy)*3 + cx;
						helper.sendBufs[bufId].push_back(p);
					}
		}
	};

	for (int i=0; i<6; i++)
		processOneSide(helper.sides[i].hostdata, helper.sides[i].size);

	MPI_Request req;
	for (int i=0; i<27; i++)
		if (i != 13 && dir2rank[i] > 0)
			logger.MPI_Check( MPI_Isend(&helper.sendBufs[i][0], helper.sendBufs[i].size(), mpiParticleType, dir2rank[i], 0, haloComm, &req) );

	printf("sending done\n");
}

void HaloExchanger::receive(int n)
{
	HaloHelper& helper = helpers[n];
	auto& pv = particleVectors[n];

	int cur = 0;
	for (int i=0; i<nActiveNeighbours; i++)
	{
		MPI_Status stat;
		int recvd = 0;
		while (recvd == 0)
		{
			logger.MPI_Check( MPI_Iprobe(MPI_ANY_SOURCE, 0, haloComm, &recvd, &stat) );
			usleep(10);
		}

		printf("");
		int msize;
	    logger.MPI_Check( MPI_Get_count(&stat, mpiParticleType, &msize) );

		pv->halo.resize(pv->halo.size + msize, resizePreserve, helper.stream);
		logger.MPI_Check( MPI_Recv(pv->halo.hostdata+cur, msize, mpiParticleType, stat.MPI_SOURCE, 0, haloComm, &stat) );
		cur += msize;
	}

	pv->halo.synchronize(synchronizeDevice, helper.stream);
}
