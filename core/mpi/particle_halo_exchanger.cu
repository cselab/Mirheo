/** @file */

#include "particle_halo_exchanger.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>
#include <core/pvs/extra_data/packers.h>

#include "valid_cell.h"

/**
 * Get halos
 * @param cinfo
 * @param packer
 * @param dataWrap
 */
template<bool QUERY=false>
__global__ void getHalos(const CellListInfo cinfo, const ParticlePacker packer, BufferOffsetsSizesWrap dataWrap)
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	int cid;
	int cx, cy, cz;

	bool valid = isValidCell(cid, cx, cy, cz, gid, blockIdx.y, cinfo);

	int pstart = valid ? cinfo.cellStarts[cid]   : 0;
	int pend   = valid ? cinfo.cellStarts[cid+1] : 0;

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
				haloOffset[current] = atomicAdd(blockSum + bufId, pend-pstart);
				current++;
			}

	__syncthreads();

	if (tid < 27 && blockSum[tid] > 0)
		blockSum[tid] = atomicAdd(dataWrap.sizes + tid, blockSum[tid]);

	if (QUERY) return;

	__syncthreads();

#pragma unroll 2
	for (int i=0; i<current; i++)
	{
		const int bufId = validHalos[i];
		const int myid  = blockSum[bufId] + haloOffset[i];

		const int ix = bufId % 3;
		const int iy = (bufId / 3) % 3;
		const int iz = bufId / 9;
		const float3 shift{ cinfo.localDomainSize.x*(ix-1),
							cinfo.localDomainSize.y*(iy-1),
							cinfo.localDomainSize.z*(iz-1) };

#pragma unroll 3
		for (int i = 0; i < pend-pstart; i++)
		{
			const int dstInd = myid   + i;
			const int srcInd = pstart + i;

			auto bufferAddr = dataWrap.buffer + dataWrap.offsets[bufId]*packer.packedSize_byte;

//			Particle p(cinfo.particles, srcInd);
//			p.r -= shift;
//			p.write2Float4((float4*)(bufferAddr), dstInd);

			packer.packShift(srcInd, bufferAddr + dstInd*packer.packedSize_byte, -shift);
		}
	}
}

__global__ static void unpackParticles(ParticlePacker packer, int startDstId, char* buffer, int np)
{
	const int pid = blockIdx.x*blockDim.x + threadIdx.x;
	if (pid >= np) return;

	packer.unpack(buffer + pid*packer.packedSize_byte, pid+startDstId);
}

//===============================================================================================
// Member functions
//===============================================================================================

bool ParticleHaloExchanger::needExchange(int id)
{
	return !particles[id]->haloValid;
}

void ParticleHaloExchanger::attach(ParticleVector* pv, CellList* cl)
{
	particles.push_back(pv);
	cellLists.push_back(cl);

	auto helper = new ExchangeHelper(pv->name, sizeof(Particle));
	helpers.push_back(helper);

	info("Particle halo exchanger takes pv %s, base tag %d", pv->name.c_str(), tagByName(pv->name));
}

void ParticleHaloExchanger::prepareData(int id, cudaStream_t stream)
{
	auto pv = particles[id];
	auto cl = cellLists[id];
	auto helper = helpers[id];

	debug2("Preparing %s halo on the device", pv->name.c_str());

	helper->sendSizes.clear(stream);
	if (pv->local()->size() > 0)
	{
		const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
		const int nthreads = 64;
		const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

		auto packer = ParticlePacker(pv, pv->local());
		helper->setDatumSize(packer.packedSize_byte);

		SAFE_KERNEL_LAUNCH(
				getHalos<true>,
				nblocks, nthreads, 0, stream,
				cl->cellInfo(), packer, helper->wrapSendData() );

		helper->makeSendOffsets_Dev2Dev(stream);
		helper->resizeSendBuf();

		helper->sendSizes.clearDevice(stream);
		SAFE_KERNEL_LAUNCH(
				getHalos<false>,
				nblocks, nthreads, 0, stream,
				cl->cellInfo(), packer, helper->wrapSendData() );
	}

	debug2("%s halo prepared", pv->name.c_str());
}

void ParticleHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
	auto pv = particles[id];
	auto helper = helpers[id];

	int totalRecvd = helper->recvOffsets[helper->nBuffers];
	pv->halo()->resize_anew(totalRecvd);

	// std::swap(pv->halo()->coosvels, helper->recvBuf);
	// TODO: types are different, cannot swap. Make consume member

//	CUDA_Check( cudaMemcpyAsync(
//			pv->halo()->coosvels.devPtr(),
//			helper->recvBuf.devPtr(),
//			helper->recvBuf.size(), cudaMemcpyDeviceToDevice, stream) );

	int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			unpackParticles,
			getNblocks(totalRecvd, nthreads), nthreads, 0, stream,
			ParticlePacker(pv, pv->halo()), 0, helper->recvBuf.devPtr(), totalRecvd );

	pv->haloValid = true;
}






