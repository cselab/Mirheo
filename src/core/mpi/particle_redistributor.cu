#include "particle_redistributor.h"
#include "exchange_helpers.h"
#include "fragments_mapping.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/extra_data/packers.h>
#include <core/utils/cuda_common.h>
#include <core/pvs/extra_data/packers.h>

#include <core/mpi/valid_cell.h>

static __device__ int encodeCellId1d(int cid, int ncells) {
    if (cid < 0)            return -1;
    else if (cid >= ncells) return 1;
    else                    return 0;
}

static __device__ int3 encodeCellId(int3 cid, int3 ncells) {
    cid.x = encodeCellId1d(cid.x, ncells.x);
    cid.y = encodeCellId1d(cid.y, ncells.y);
    cid.z = encodeCellId1d(cid.z, ncells.z);
    return cid;
}

static __device__ bool hasToLeave(int3 dir) {
    return dir.x != 0 || dir.y != 0 || dir.z != 0;
}

enum class PackMode
{
    Querry, Pack
};

template <PackMode packMode>
__global__ void getExitingParticles(const CellListInfo cinfo, ParticlePacker packer, BufferOffsetsSizesWrap dataWrap)
{
    const int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int cid;
    int dx, dy, dz;
    const int3 ncells = cinfo.ncells;

    bool valid = isValidCell(cid, dx, dy, dz, gid, blockIdx.y, cinfo);

    if (!valid) return;

    // The following is called for every outer cell and exactly once for each
    //
    // Now for each cell we check its every particle if it needs to move

    int pstart = cinfo.cellStarts[cid];
    int pend   = cinfo.cellStarts[cid+1];

#pragma unroll 2
    for (int i = 0; i < pend-pstart; i++)
    {
        const int srcId = pstart + i;
        Particle p(cinfo.particles, srcId);

        int3 dir = cinfo.getCellIdAlongAxes<CellListsProjection::NoClamp>(make_float3(p.r));

        dir = encodeCellId(dir, ncells);

        if (p.isMarked()) continue;
        
        if (hasToLeave(dir)) {
            const int bufId = FragmentMapping::getId(dir);
            const float3 shift{ cinfo.localDomainSize.x * dir.x,
                                cinfo.localDomainSize.y * dir.y,
                                cinfo.localDomainSize.z * dir.z };

            int myid = atomicAdd(dataWrap.sizes + bufId, 1);

            if (packMode == PackMode::Querry) {
                continue;
            }
            else {
                auto bufferAddr = dataWrap.buffer + dataWrap.offsets[bufId]*packer.packedSize_byte;
                packer.packShift(srcId, bufferAddr + myid*packer.packedSize_byte, -shift);

                // mark the particle as exited to assist cell-list building
                Float3_int pos = p.r2Float3_int();
                pos.mark();
                cinfo.particles[2*srcId] = pos.toFloat4();
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

bool ParticleRedistributor::needExchange(int id)
{
    return !particles[id]->redistValid;
}

void ParticleRedistributor::attach(ParticleVector* pv, CellList* cl)
{
    particles.push_back(pv);
    cellLists.push_back(cl);

    if (dynamic_cast<PrimaryCellList*>(cl) == nullptr)
        die("Redistributor (for %s) should be used with the primary cell-lists only!", pv->name.c_str());

    auto helper = new ExchangeHelper(pv->name, sizeof(Particle));
    helpers.push_back(helper);

    info("Particle redistributor takes pv '%s'", pv->name.c_str());
}

void ParticleRedistributor::prepareSizes(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id];

    debug2("Counting leaving particles of '%s'", pv->name.c_str());

    helper->sendSizes.clear(stream);
    if (pv->local()->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

        auto packer = ParticlePacker(pv, pv->local(), stream);
        helper->setDatumSize(packer.packedSize_byte);

        SAFE_KERNEL_LAUNCH(
                getExitingParticles<PackMode::Querry>,
                nblocks, nthreads, 0, stream,
                cl->cellInfo(), packer, helper->wrapSendData() );

        helper->computeSendOffsets_Dev2Dev(stream);
    }
}

void ParticleRedistributor::prepareData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto cl = cellLists[id];
    auto helper = helpers[id];

    debug2("Downloading %d leaving particles of '%s'",
           helper->sendOffsets[FragmentMapping::numFragments], pv->name.c_str());

    if (pv->local()->size() > 0)
    {
        const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
        const int nthreads = 64;
        const dim3 nblocks = dim3(getNblocks(maxdim*maxdim, nthreads), 6, 1);

        auto packer = ParticlePacker(pv, pv->local(), stream);

        helper->resizeSendBuf();
        // Sizes will still remain on host, no need to download again
        helper->sendSizes.clearDevice(stream);
        SAFE_KERNEL_LAUNCH(
                getExitingParticles<PackMode::Pack>,
                nblocks, nthreads, 0, stream,
                cl->cellInfo(), packer, helper->wrapSendData() );
    }
}

void ParticleRedistributor::combineAndUploadData(int id, cudaStream_t stream)
{
    auto pv = particles[id];
    auto helper = helpers[id];

    int oldsize = pv->local()->size();
    int totalRecvd = helper->recvOffsets[helper->nBuffers];
    pv->local()->resize(oldsize + totalRecvd,  stream);

    if (totalRecvd > 0)
    {
        int nthreads = 64;
        SAFE_KERNEL_LAUNCH(
                unpackParticles,
                getNblocks(totalRecvd, nthreads), nthreads, 0, stream,
                ParticlePacker(pv, pv->local(), stream), oldsize, helper->recvBuf.devPtr(), totalRecvd );

//        CUDA_Check( cudaMemcpyAsync(
//                pv->local()->coosvels.devPtr() + oldsize,
//                helper->recvBuf.devPtr(),
//                helper->recvBuf.size(), cudaMemcpyDeviceToDevice, stream) );
    }

    pv->redistValid = true;

    // Particles may have migrated, rebuild cell-lists
    if (totalRecvd > 0)    pv->cellListStamp++;
}
