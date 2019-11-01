#include "particle_halo_exchanger.h"

#include "exchange_helpers.h"
#include "utils/common.h"
#include "utils/face_dispatch.h"
#include "utils/fragments_mapping.h"
#include "utils/map.h"

#include <core/celllist.h>
#include <core/logger.h>
#include <core/pvs/packers/particles.h>
#include <core/pvs/particle_vector.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <unistd.h>

enum class PackMode
{
    Query, Pack
};

namespace ParticleHaloExchangersKernels
{

template <PackMode packMode>
__global__ void getHalo(const CellListInfo cinfo, DomainInfo domain,
                        ParticlePackerHandler packer, BufferOffsetsSizesWrap dataWrap)
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    const int faceId = blockIdx.y;
    int cid;
    int dx, dy, dz;

    const bool valid = distributeThreadsToFaceCell(cid, dx, dy, dz, gid, faceId, cinfo);

    const int pstart = valid ? cinfo.cellStarts[cid]   : 0;
    const int pend   = valid ? cinfo.cellStarts[cid+1] : 0;

    // Use shared memory to decrease number of global atomics
    // We're sending to max 7 halos (corner)
    char validHalos[7];
    int haloOffset[7] = {};

    int current = 0;

    // Total number of elements written to halos by this block
    __shared__ int blockSum[FragmentMapping::numFragments];
    if (tid < FragmentMapping::numFragments) blockSum[tid] = 0;

    __syncthreads();

    for (int ix = math::min(dx, 0); ix <= math::max(dx, 0); ix++)
        for (int iy = math::min(dy, 0); iy <= math::max(dy, 0); iy++)
            for (int iz = math::min(dz, 0); iz <= math::max(dz, 0); iz++)
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

    if (packMode == PackMode::Query)
    {
        return;
    }
    else
    {
        __syncthreads();

#pragma unroll 2
        for (int j = 0; j < current; ++j)
        {
            const int bufId = validHalos[j];
            const int myId  = blockSum[bufId] + haloOffset[j];

            auto dir = FragmentMapping::getDir(bufId);
            auto shift = ExchangersCommon::getShift(domain.localSize, dir);

            const int numElements = dataWrap.offsets[bufId+1] - dataWrap.offsets[bufId];
            auto buffer = dataWrap.getBuffer(bufId);
            
#pragma unroll 3
            for (int i = 0; i < pend-pstart; ++i)
            {
                const int dstPid = myId   + i;
                const int srcPid = pstart + i;

                packer.particles.packShift(srcPid, dstPid, buffer, numElements, shift);
            }
        }
    }
}

__global__ void unpackParticles(BufferOffsetsSizesWrap dataWrap, ParticlePackerHandler packer)
{
    const int tid = threadIdx.x;
    const int pid = tid + blockIdx.x * blockDim.x;
    
    extern __shared__ int offsets[];
    
    const int nBuffers = dataWrap.nBuffers;

    for (int i = tid; i < nBuffers + 1; i += blockDim.x)
        offsets[i] = dataWrap.offsets[i];
    __syncthreads();

    if (pid >= offsets[nBuffers]) return;

    const int bufId = dispatchThreadsPerBuffer(nBuffers, offsets, pid);
    
    auto buffer = dataWrap.getBuffer(bufId);
    const int numElements = dataWrap.sizes[bufId];
    
    const int srcPid = pid - offsets[bufId];
    const int dstPid = pid;

    packer.particles.unpack(srcPid, dstPid, buffer, numElements);
}

} //namespace ParticleHaloExchangersKernels


//===============================================================================================
// Member functions
//===============================================================================================

ParticleHaloExchanger::ParticleHaloExchanger() = default;
ParticleHaloExchanger::~ParticleHaloExchanger() = default;

void ParticleHaloExchanger::attach(ParticleVector *pv, CellList *cl, const std::vector<std::string>& extraChannelNames)
{
    const int id = particles.size();
    particles.push_back(pv);
    cellLists.push_back(cl);

    auto channels = extraChannelNames;
    channels.push_back(ChannelNames::positions);
    channels.push_back(ChannelNames::velocities);

    PackPredicate predicate = [channels](const DataManager::NamedChannelDesc& namedDesc)
    {
        return std::find(channels.begin(), channels.end(), namedDesc.first) != channels.end();
    };
    
    auto   packer = std::make_unique<ParticlePacker> (predicate);
    auto unpacker = std::make_unique<ParticlePacker> (predicate);
    auto   helper = std::make_unique<ExchangeHelper> (pv->name, id, packer.get());
    
    helpers  .push_back(std::move(  helper));
    packers  .push_back(std::move(  packer));
    unpackers.push_back(std::move(unpacker));
    
    std::string msg_channels = channels.empty() ? "no channels." : "with channels: ";
    for (const auto& ch : channels) msg_channels += "'" + ch + "' ";
    
    info("Particle halo exchanger takes pv '%s' with celllist of rc = %g, %s",
         pv->name.c_str(), cl->rc, msg_channels.c_str());
}

void ParticleHaloExchanger::prepareSizes(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id].get();
    auto packer = packers[id].get();

    debug2("Counting halo particles of '%s'", pv->name.c_str());

    LocalParticleVector *lpv = cl->getLocalParticleVector();
    
    helper->send.sizes.clearDevice(stream);
    packer->update(lpv, stream);
    
    if (lpv->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const int nfaces   = 6;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), nfaces, 1);

        SAFE_KERNEL_LAUNCH(
            ParticleHaloExchangersKernels::getHalo<PackMode::Query>,
            nblocks, nthreads, 0, stream,
            cl->cellInfo(), pv->state->domain,
            packer->handler(), helper->wrapSendData() );
    }

    helper->computeSendOffsets_Dev2Dev(stream);
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
        const int nfaces   = 6;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), nfaces, 1);

        helper->resizeSendBuf();
        helper->send.sizes.clearDevice(stream);
        
        SAFE_KERNEL_LAUNCH(
            ParticleHaloExchangersKernels::getHalo<PackMode::Pack>,
            nblocks, nthreads, 0, stream,
            cl->cellInfo(), pv->state->domain,
            packer->handler(), helper->wrapSendData() );
    }
}

void ParticleHaloExchanger::combineAndUploadData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto helper   = helpers  [id].get();
    auto unpacker = unpackers[id].get();

    auto lpv = pv->halo();

    const auto& offsets = helper->recv.offsets;
    
    int totalRecvd = offsets[helper->nBuffers];
    lpv->resize_anew(totalRecvd);

    debug2("received %d particles from halo exchange", totalRecvd);

    unpacker->update(lpv, stream);
    
    const int nthreads = 128;
    const int nblocks  = getNblocks(totalRecvd, nthreads);
    const size_t shMemSize = offsets.size() * sizeof(offsets[0]);
    
    SAFE_KERNEL_LAUNCH(
        ParticleHaloExchangersKernels::unpackParticles,
        nblocks, nthreads, shMemSize, stream,
        helper->wrapRecvData(), unpacker->handler());
    
    pv->haloValid = true;
}

bool ParticleHaloExchanger::needExchange(int id)
{
    return !particles[id]->haloValid;
}
