#include "particle_halo_exchanger.h"
#include "exchange_helpers.h"
#include "packers/particles.h"
#include "utils/fragments_mapping.h"
#include "utils/face_dispatch.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>
#include <core/pvs/extra_data/packers.h>

#include <unistd.h>

enum class PackMode
{
    Query, Pack
};

namespace ParticleHaloExchangersKernels
{

/**
 * Get halos
 * @param cinfo
 * @param packer
 * @param dataWrap
 */
template <PackMode packMode>
__global__ void getHaloMap(const CellListInfo cinfo, MapEntry *map, BufferOffsetsSizesWrap dataWrap)
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int faceId = blockIdx.y;
    int cid;
    int dx, dy, dz;

    bool valid = distributeThreadsToFaceCell(cid, dx, dy, dz, gid, faceId, cinfo);

    int pstart = valid ? cinfo.cellStarts[cid]   : 0;
    int pend   = valid ? cinfo.cellStarts[cid+1] : 0;

    // Use shared memory to decrease number of global atomics
    // We're sending to max 7 halos (corner)
    char validHalos[7];
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
            const int myId  = blockSum[bufId] + haloOffset[i];

#pragma unroll 3
            for (int i = 0; i < pend-pstart; i++)
            {
                const int dstId = myId   + i;
                const int srcId = pstart + i;

                int offset = dataWrap.offsets[bufId];
                map[offset + dstId] = MapEntry(srcId, bufId);
            }
        }
    }
}

} //namespace ParticleHaloExchangersKernels


//===============================================================================================
// Member functions
//===============================================================================================

ParticleHaloExchanger::ParticleHaloExchanger() = default;
ParticleHaloExchanger::~ParticleHaloExchanger() = default;

void ParticleHaloExchanger::attach(ParticleVector *pv, CellList *cl, const std::vector<std::string>& extraChannelNames)
{
    int id = particles.size();
    particles.push_back(pv);
    cellLists.push_back(cl);

    auto packer = std::make_unique<ParticlesPacker> (pv, [extraChannelNames](const DataManager::NamedChannelDesc& namedDesc) {
        return std::find(extraChannelNames.begin(), extraChannelNames.end(), namedDesc.first) != extraChannelNames.end();
    });
    auto helper = std::make_unique<ExchangeHelper> (pv->name, id, packer.get());

    helpers.push_back(std::move(helper));
    packers.push_back(std::move(packer));

    std::string msg_channels = extraChannelNames.empty() ? "no extra channels." : "with extra channels: ";
    for (const auto& ch : extraChannelNames) msg_channels += "'" + ch + "' ";
    
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
    
    helper->send.sizes.clear(stream);

    if (lpv->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});

        const int nthreads = 64;
        const int nfaces = 6;;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), nfaces, 1);

        SAFE_KERNEL_LAUNCH(
            ParticleHaloExchangersKernels::getHaloMap<PackMode::Query>,
            nblocks, nthreads, 0, stream,
            cl->cellInfo(), nullptr, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
}

void ParticleHaloExchanger::prepareData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int nEntities = helper->send.offsets[helper->nBuffers];
    
    debug2("Downloading %d halo particles of '%s'", nEntities, pv->name.c_str());

    LocalParticleVector *lpv = cl->getLocalParticleVector();

    if (lpv->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const int nfaces = 6;;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), nfaces, 1);

        helper->resizeSendBuf();
        helper->map.resize_anew(nEntities);
        helper->send.sizes.clearDevice(stream);
        
        SAFE_KERNEL_LAUNCH(
            ParticleHaloExchangersKernels::getHaloMap<PackMode::Pack>,
            nblocks, nthreads, 0, stream,
            cl->cellInfo(), helper->map.devPtr(), helper->wrapSendData() );

        packer->packToBuffer(lpv, helper->map, &helper->send, {}, stream);
    }
}

void ParticleHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    int totalRecvd = helper->recv.offsets[helper->nBuffers];
    pv->halo()->resize_anew(totalRecvd);

    debug2("received %d particles from halo exchange", totalRecvd);

    if (totalRecvd > 0)
        packer->unpackFromBuffer(pv->halo(), &helper->recv, 0, stream);
    
    pv->haloValid = true;
}

bool ParticleHaloExchanger::needExchange(int id)
{
    return !particles[id]->haloValid;
}
