#include "impose_profile.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

__device__ inline bool all_lt(real3 a, real3 b)
{
    return a.x < b.x && a.y < b.y && a.z < b.z;
}

__global__ void applyProfile(
        CellListInfo cinfo, PVview view,
        const int* relevantCells, const int nRelevantCells,
        real3 low, real3 high,
        real3 targetVel,
        real kBT, real invMass, real seed1, real seed2)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nRelevantCells) return;

    int pstart = cinfo.cellStarts[ relevantCells[gid]   ];
    int pend   = cinfo.cellStarts[ relevantCells[gid]+1 ];

#pragma unroll 3
    for (int pid = pstart; pid < pend; pid++)
    {
        Particle p(view.readParticle(pid));

        if (all_lt(low, p.r) && all_lt(p.r, high))
        {
            real2 rand1 = Saru::normal2(seed1 + pid, threadIdx.x, blockIdx.x);
            real2 rand2 = Saru::normal2(seed2 + pid, threadIdx.x, blockIdx.x);

            p.u = targetVel + math::sqrt(kBT * invMass) * make_real3(rand1.x, rand1.y, rand2.x);
            view.writeParticle(pid, p);
        }
    }
}

template<bool QUERY>
__global__ void getRelevantCells(
        CellListInfo cinfo,
        real3 low, real3 high,
        int* relevantCells, int* nRelevantCells)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= cinfo.totcells) return;

    int3 ind;
    cinfo.decode(cid, ind.x, ind.y, ind.z);
    real3 botCell = -0.5_r * cinfo.localDomainSize + make_real3(ind)*cinfo.h;
    real3 topCell = botCell + cinfo.h;

    bool relevant = all_lt(low, topCell) && all_lt(botCell, high);

    if (relevant)
    {
        int id = atomicAggInc(nRelevantCells);
        if (!QUERY) relevantCells[id] = cid;
    }
}

ImposeProfilePlugin::ImposeProfilePlugin(const MirState *state, std::string name, std::string pvName,
                                         real3 low, real3 high, real3 targetVel, real kBT) :
    SimulationPlugin(state, name), pvName(pvName), low(low), high(high), targetVel(targetVel), kBT(kBT)
{}

void ImposeProfilePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
    cl = simulation->gelCellList(pv);

    if (cl == nullptr)
        die("Cell-list is required for PV '%s' by plugin '%s'", pvName.c_str(), name.c_str());

    debug("Setting up pluging '%s' to impose uniform profile with velocity [%f %f %f]"
          " and temperature %f in a box [%.2f %.2f %.2f] - [%.2f %.2f %.2f] for PV '%s'",
          name.c_str(), targetVel.x, targetVel.y, targetVel.z, kBT,
          low.x, low.y, low.z, high.x, high.y, high.z, pv->name.c_str());

    low  = getState()->domain.global2local(low);
    high = getState()->domain.global2local(high);

    const int nthreads = 128;

    nRelevantCells.clearDevice(0);
    SAFE_KERNEL_LAUNCH(
            getRelevantCells<true>,
            getNblocks(cl->totcells, nthreads), nthreads, 0, 0,
            cl->cellInfo(), low, high, relevantCells.devPtr(), nRelevantCells.devPtr() );

    nRelevantCells.downloadFromDevice(0);
    relevantCells.resize_anew(nRelevantCells[0]);
    nRelevantCells.clearDevice(0);

    SAFE_KERNEL_LAUNCH(
            getRelevantCells<false>,
            getNblocks(cl->totcells, nthreads), nthreads, 0, 0,
            cl->cellInfo(), low, high, relevantCells.devPtr(), nRelevantCells.devPtr() );
}

void ImposeProfilePlugin::afterIntegration(cudaStream_t stream)
{
    const int nthreads = 128;

    debug2("Imposing uniform profile for PV '%s' as per plugin '%s'",
           pv->name.c_str(), name.c_str());

    SAFE_KERNEL_LAUNCH(
            applyProfile,
            getNblocks(nRelevantCells[0], nthreads), nthreads, 0, stream,

            cl->cellInfo(), cl->getView<PVview>(), relevantCells.devPtr(), nRelevantCells[0], low, high, targetVel,
            kBT, 1.0_r / pv->mass, drand48(), drand48() );
}

} // namespace mirheo
