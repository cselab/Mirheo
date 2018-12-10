#include "particle_halo_exchanger.h"
#include "exchange_helpers.h"
#include "fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>
#include <core/pvs/extra_data/packers.h>

#include <unistd.h>


#include "valid_cell.h"

enum class PackMode
{
    Querry, Pack
};

/**
 * Get halos
 * @param cinfo
 * @param packer
 * @param dataWrap
 */
template <PackMode packMode>
__global__ void getHalos(const CellListInfo cinfo, const ParticlePacker packer, BufferOffsetsSizesWrap dataWrap)
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    int cid;
    int dx, dy, dz;

    bool valid = isValidCell(cid, dx, dy, dz, gid, blockIdx.y, cinfo);

    int pstart = valid ? cinfo.cellStarts[cid]   : 0;
    int pend   = valid ? cinfo.cellStarts[cid+1] : 0;

    // Use shared memory to decrease number of global atomics
    // We're sending to max 7 halos (corner)
    short validHalos[7];
    int haloOffset[7] = {};

    int current = 0;

    // Total number of elements written to halos by this block
    __shared__ int blockSum[FragmentMapping::numFragments];
    if (tid < FragmentMapping::numFragments) blockSum[tid] = 0;

    __syncthreads();

    for (int ix = min(dx, 0); ix <= max(dx, 0); ix++)
        for (int iy = min(dy, 0); iy <= max(dy, 0); iy++)
            for (int iz = min(dz, 0); iz <= max(dz, 0); iz++)
            {
                if (ix == 0 && iy == 0 && iz == 0) continue;

                const int bufId = FragmentMapping::getId(ix, iy, iz);
                validHalos[current] = bufId;
                haloOffset[current] = atomicAdd(blockSum + bufId, pend-pstart);
                current++;
            }

    __syncthreads();

    if (tid < FragmentMapping::numFragments && blockSum[tid] > 0)
        blockSum[tid] = atomicAdd(dataWrap.sizes + tid, blockSum[tid]);

    if (packMode == PackMode::Querry) {
        return;
    }
    else {
        __syncthreads();

#pragma unroll 2
        for (int i=0; i<current; i++)
        {
            const int bufId = validHalos[i];
            const int myid  = blockSum[bufId] + haloOffset[i];

            const int3 dir = FragmentMapping::getDir(bufId);

            const float3 shift{ cinfo.localDomainSize.x * dir.x,
                                cinfo.localDomainSize.y * dir.y,
                                cinfo.localDomainSize.z * dir.z };

#pragma unroll 3
            for (int i = 0; i < pend-pstart; i++)
            {
                const int dstInd = myid   + i;
                const int srcInd = pstart + i;

                auto bufferAddr = dataWrap.buffer + dataWrap.offsets[bufId]*packer.packedSize_byte;

                packer.packShift(srcInd, bufferAddr + dstInd*packer.packedSize_byte, -shift);
            }
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

    info("Particle halo exchanger takes pv '%s'", pv->name.c_str());
}

void ParticleHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id];

    debug2("Counting halo particles of '%s'", pv->name.c_str());

    helper->sendSizes.clear(stream);
    if (pv->local()->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

        auto packer = ParticlePacker(pv, pv->local(), stream);
        helper->setDatumSize(packer.packedSize_byte);

        SAFE_KERNEL_LAUNCH(
                getHalos<PackMode::Querry>,
                nblocks, nthreads, 0, stream,
                cl->cellInfo(), packer, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
}

void ParticleHaloExchanger::prepareData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id];

    debug2("Downloading %d halo particles of '%s'",
           helper->sendOffsets[FragmentMapping::numFragments], pv->name.c_str());

    if (pv->local()->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

        auto packer = ParticlePacker(pv, pv->local(), stream);

        helper->resizeSendBuf();
        helper->sendSizes.clearDevice(stream);
        SAFE_KERNEL_LAUNCH(
                getHalos<PackMode::Pack>,
                nblocks, nthreads, 0, stream,
                cl->cellInfo(), packer, helper->wrapSendData() );
    }
}

void ParticleHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto helper = helpers[id];

    int totalRecvd = helper->recvOffsets[helper->nBuffers];
    pv->halo()->resize_anew(totalRecvd);

    int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            unpackParticles,
            getNblocks(totalRecvd, nthreads), nthreads, 0, stream,
            ParticlePacker(pv, pv->halo(), stream), 0, helper->recvBuf.devPtr(), totalRecvd );

    pv->haloValid = true;
}






