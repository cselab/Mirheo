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
    Query, Pack
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

    if (packMode == PackMode::Query) {
        return;
    }
    else {
        __syncthreads();

#pragma unroll 2
        for (int i = 0; i < current; i++)
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

__global__ static void unpackParticles(ParticlePacker packer, const char *buffer, int np)
{
    const int pid = blockIdx.x*blockDim.x + threadIdx.x;
    if (pid >= np) return;

    packer.unpack(buffer + pid*packer.packedSize_byte, pid);
}


//===============================================================================================
// Member functions
//===============================================================================================

ParticleHaloExchanger::~ParticleHaloExchanger() = default;

void ParticleHaloExchanger::attach(ParticleVector *pv, CellList *cl, const std::vector<std::string>& extraChannelNames)
{
    int id = particles.size();
    particles.push_back(pv);
    cellLists.push_back(cl);

    auto helper = std::make_unique<ExchangeHelper> (pv->name, id);
    helper->setDatumSize(sizeof(Particle));

    helpers.push_back(std::move(helper));

    packPredicates.push_back([extraChannelNames](const ExtraDataManager::NamedChannelDesc& namedDesc) {
        return std::find(extraChannelNames.begin(), extraChannelNames.end(), namedDesc.first) != extraChannelNames.end();
    });

    std::string msg_channels = extraChannelNames.empty() ?
        "no extra channels." :
        "with extra channels: ";
    for (const auto& ch : extraChannelNames)
        msg_channels += "'" + ch + "' ";
    
    info("Particle halo exchanger takes pv '%s' with celllist of rc = %g, %s",
         pv->name.c_str(), cl->rc, msg_channels.c_str());
}

void ParticleHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id].get();

    debug2("Counting halo particles of '%s'", pv->name.c_str());

    LocalParticleVector *lpv = cl->getLocalParticleVector();
    // LocalParticleVector *lpv = pv->local();
    
    helper->sendSizes.clear(stream);
    if (lpv->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});

        const int nthreads = 64;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

        ParticlePacker packer(pv, lpv, packPredicates[id], stream);

        helper->setDatumSize(packer.packedSize_byte);

        SAFE_KERNEL_LAUNCH(
                getHalos<PackMode::Query>,
                nblocks, nthreads, 0, stream,
                cl->cellInfo(), packer, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
}

void ParticleHaloExchanger::prepareData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id].get();

    debug2("Downloading %d halo particles of '%s'",
           helper->sendOffsets[FragmentMapping::numFragments], pv->name.c_str());

    LocalParticleVector *lpv = cl->getLocalParticleVector();
    // LocalParticleVector *lpv = pv->local();

    if (lpv->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

        ParticlePacker packer(pv, lpv, packPredicates[id], stream);

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
    auto helper = helpers[id].get();

    int totalRecvd = helper->recvOffsets[helper->nBuffers];
    pv->halo()->resize_anew(totalRecvd);

    debug2("received %d particles from halo exchange", totalRecvd);

    ParticlePacker packer(pv, pv->halo(), packPredicates[id], stream);
    
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            unpackParticles,
            getNblocks(totalRecvd, nthreads), nthreads, 0, stream,
            packer, helper->recvBuf.devPtr(), totalRecvd );

    pv->haloValid = true;
}

bool ParticleHaloExchanger::needExchange(int id)
{
    return !particles[id]->haloValid;
}
