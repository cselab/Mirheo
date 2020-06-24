// Copyright 2020 ETH Zurich. All Rights Reserved.
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
    SimulationPlugin(state, name),
    pvName_(pvName),
    low_(low),
    high_(high),
    targetVel_(targetVel),
    kBT_(kBT)
{}

void ImposeProfilePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv_ = simulation->getPVbyNameOrDie(pvName_);
    cl_ = simulation->gelCellList(pv_);

    if (cl_ == nullptr)
        die("Cell-list is required for PV '%s' by plugin '%s'", pvName_.c_str(), getCName());

    debug("Setting up pluging '%s' to impose uniform profile with velocity [%f %f %f]"
          " and temperature %f in a box [%.2f %.2f %.2f] - [%.2f %.2f %.2f] for PV '%s'",
          getCName(), targetVel_.x, targetVel_.y, targetVel_.z, kBT_,
          low_.x, low_.y, low_.z, high_.x, high_.y, high_.z, pv_->getCName());

    low_  = getState()->domain.global2local(low_);
    high_ = getState()->domain.global2local(high_);

    const int nthreads = 128;

    nRelevantCells_.clearDevice(defaultStream);
    SAFE_KERNEL_LAUNCH(
            getRelevantCells<true>,
            getNblocks(cl_->totcells, nthreads), nthreads, 0, defaultStream,
            cl_->cellInfo(), low_, high_, relevantCells_.devPtr(), nRelevantCells_.devPtr() );

    nRelevantCells_.downloadFromDevice(defaultStream);
    relevantCells_.resize_anew(nRelevantCells_[0]);
    nRelevantCells_.clearDevice(defaultStream);

    SAFE_KERNEL_LAUNCH(
            getRelevantCells<false>,
            getNblocks(cl_->totcells, nthreads), nthreads, 0, defaultStream,
            cl_->cellInfo(), low_, high_, relevantCells_.devPtr(), nRelevantCells_.devPtr() );
}

void ImposeProfilePlugin::afterIntegration(cudaStream_t stream)
{
    const int nthreads = 128;

    debug2("Imposing uniform profile for PV '%s' as per plugin '%s'",
           pv_->getCName(), getCName());

    SAFE_KERNEL_LAUNCH(
            applyProfile,
            getNblocks(nRelevantCells_[0], nthreads), nthreads, 0, stream,
            cl_->cellInfo(), cl_->getView<PVview>(), relevantCells_.devPtr(), nRelevantCells_[0], low_, high_, targetVel_,
            kBT_, 1.0_r / pv_->getMassPerParticle(), drand48(), drand48() );
}

} // namespace mirheo
