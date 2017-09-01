#include <core/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/cuda_common.h>

#include <core/mpi/particle_halo_exchanger.h>
#include <core/mpi/valid_cell.h>

#include <algorithm>

__global__ void getHalos(const float4* __restrict__ coosvels, const CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		const int64_t dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	int cid;
	int cx, cy, cz;

	bool valid = isValidCell(cid, cx, cy, cz, gid, blockIdx.y, cinfo);

	if (__all(!valid) && tid > 27) return;

	int2 start_size = valid ? cinfo.decodeStartSize(cellsStartSize[cid]) : make_int2(0, 0);

	// Use shared memory to decrease number of global atomics
	// We're sending to max 7 halos (corner)
	short validHalos[7];
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
		const float4 shift{ cinfo.domainSize.x*(ix-1),
							cinfo.domainSize.y*(iy-1),
							cinfo.domainSize.z*(iz-1), 0.0f };

#pragma unroll 2
		for (int i = 0; i < start_size.y; i++)
		{
			const int dstInd = 2*(myid         + i);
			const int srcInd = 2*(start_size.x + i);

			float4 tmp1 = coosvels[srcInd] - shift;
			float4 tmp2 = coosvels[srcInd+1];

			float4* addr = (float4*)dests  [bufId];
			addr[dstInd + 0] = tmp1;
			addr[dstInd + 1] = tmp2;
		}
	}
}

void ParticleHaloExchanger::attach(ParticleVector* pv, CellList* cl)
{
	particles.push_back(pv);
	cellLists.push_back(cl);

	// TODO: change ndens
	const double ndens = 16;//(double)pv->local()->size() / (cl->ncells.x * cl->ncells.y * cl->ncells.z * cl->rc*cl->rc*cl->rc);
	const int maxdim = std::max({cl->domainSize.x, cl->domainSize.y, cl->domainSize.z});

	// Sizes of buffers. 0 is side, 1 is edge, 2 is corner
	const int sizes[3] = { (int)(4*ndens * maxdim*maxdim + 128), (int)(4*ndens * maxdim + 128), (int)(4*ndens + 128) };

	auto helper = new ExchangeHelper(pv->name, sizeof(Particle), sizes);
	helpers.push_back(helper);
}

void ParticleHaloExchanger::combineAndUploadData(int id)
{
	auto pv = particles[id];
	auto helper = helpers[id];

	pv->halo()->resize(helper->recvOffsets[27], helper->stream, ResizeKind::resizeAnew);

	for (int i=0; i < 27; i++)
	{
		const int msize = helper->recvOffsets[i+1] - helper->recvOffsets[i];
		if (msize > 0)
			CUDA_Check( cudaMemcpyAsync(pv->halo()->coosvels.devPtr() + helper->recvOffsets[i], helper->recvBufs[i].hostPtr(),
					msize*sizeof(Particle), cudaMemcpyHostToDevice, helper->stream) );
	}
}

void ParticleHaloExchanger::prepareData(int id, cudaStream_t defStream)
{
	auto pv = particles[id];
	auto cl = cellLists[id];
	auto helper = helpers[id];

	debug2("Preparing %s halo on the device", pv->name.c_str());

	helper->bufSizes.clearDevice(defStream);

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
	const int nthreads = 32;
	if (pv->local()->size() > 0)
		getHalos<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, defStream >>>
				((float4*)pv->local()->coosvels.devPtr(), cl->cellInfo(), cl->cellsStartSize.devPtr(), (int64_t*)helper->sendAddrs.devPtr(), helper->bufSizes.devPtr());
}

