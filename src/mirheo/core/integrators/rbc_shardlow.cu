// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "rbc_shardlow.h"

#include <mirheo/core/interactions/membrane/base_membrane.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/membrane_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <memory>

namespace mirheo
{

namespace rbc_shardlow_kernels {

__global__ void velocityVerletStep1(PVview view, const real dt_2m, const real dt)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= view.size)
        return;

    auto r = Real3_int( view.positions[i] );
    auto v = Real3_int( view.velocities[i] );
    const real3 f = Force(view.forces[i]).f;

    v.v += dt_2m * f;
    r.v += dt * v.v;

    view.positions[i] = r.toReal4();
    view.velocities[i] = v.toReal4();
}

__global__ void velocityVerletStep2(PVview view, const real dt_2m)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= view.size)
        return;

    auto v = Real3_int( view.velocities[i] );
    const real3 f = Force(view.forces[i]).f;

    v.v += dt_2m * f;

    view.velocities[i] = v.toReal4();
}

__global__ void sweepVelocities(int nEdges, const int2 *edges, OVview view,
                                real dt, real gamma, real sigma, real invMass, real seed)
{
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;

    const int rbcId  = gid / nEdges;
    const int edgeId = gid % nEdges;

    if (rbcId >= view.nObjects)
        return;

    const real startId = rbcId * view.objSize;

    const int i = startId + edges[edgeId].x;
    const int j = startId + edges[edgeId].y;


    const real3 ri = Real3_int( view.positions[i] ).v;
    const real3 rj = Real3_int( view.positions[j] ).v;

    Real3_int vi ( view.velocities[i] );
    Real3_int vj ( view.velocities[j] );

    const real3 eij = normalize(rj - ri);

    const real inv2m = 0.5_r * invMass;
    const real dtgamma = dt * gamma;
    const real sqrtdtsigma = math::sqrt(dt) * sigma;

    // step 1: explicit
    {
        const real3 vij = vj.v - vi.v;
        constexpr real sqrt_12 = 3.4641016151_r;
        const real xiij = sqrt_12 * (Saru::uniform01(seed, i, j) - 0.5_r);

        const real3 dv = (dtgamma * inv2m * dot(eij, vij) -
                          sqrtdtsigma * inv2m * xiij) * eij;

        vi.v += dv;
        vj.v -= dv;
    }

    // step 2: implicit
    {
        const real3 vij = vj.v - vi.v;
        constexpr real sqrt_12 = 3.4641016151_r;
        const real xiij = sqrt_12 * (Saru::uniform01(seed, i, j) - 0.5_r);

        const real frac = inv2m * dtgamma / (1.0_r + dtgamma);

        const real3 dv = (frac * (dot(eij, vij) + sqrtdtsigma * xiij)
                          - inv2m * sqrtdtsigma * xiij) * eij;

        vi.v += dv;
        vj.v -= dv;
    }

    view.velocities[i] = vi.toReal4();
    view.velocities[j] = vj.toReal4();
}

} // namespace rbc_shardlow_kernels


IntegratorSubStepShardlowSweep::IntegratorSubStepShardlowSweep(const MirState *state, const std::string& name, int substeps,
                                                               BaseMembraneInteraction* fastForces,
                                                               real gammaC, real kBT, int nsweeps) :

    Integrator(state, name),
    substeps_(substeps),
    fastForces_(fastForces),
    subState_(*state),
    gammaC_(gammaC),
    kBT_(kBT),
    nsweeps_(nsweeps)
{
    debug("setup substep integrator '%s' for %d substeps with %d sweeps",
          getCName(), substeps_, nsweeps_);
}

IntegratorSubStepShardlowSweep::~IntegratorSubStepShardlowSweep() = default;

void IntegratorSubStepShardlowSweep::execute(ParticleVector *pv, cudaStream_t stream)
{
    auto *mv = dynamic_cast<MembraneVector*>(pv);

    if (nullptr == mv)
        die("'%s' expects a MembraneVector, got '%s'",
            getCName(), pv->getCName());

    // save "slow forces"
    slowForces_.copyFromDevice(pv->local()->forces(), stream);

    // initialize the forces for the first half step
    fastForces_->local(pv, pv, nullptr, nullptr, stream);

    // save previous positions
    previousPositions_.copyFromDevice(pv->local()->positions(), stream);

    PVview pvView(pv, pv->local());

    const real dt = getState()->getDt() / substeps_;
    const real dt_2m = 0.5_r * dt / pvView.mass;

    for (int substep = 0; substep < substeps_; ++substep)
    {
        _viscousSweeps(mv, stream);

        constexpr int nthreads = 128;
        const int nblocks = getNblocks(pvView.size, nthreads);

        SAFE_KERNEL_LAUNCH(
            rbc_shardlow_kernels::velocityVerletStep1,
            nblocks, nthreads, 0, stream,
            pvView, dt_2m, dt);

        pv->local()->forces().copy(slowForces_, stream);
        fastForces_->local(pv, pv, nullptr, nullptr, stream);

        SAFE_KERNEL_LAUNCH(
            rbc_shardlow_kernels::velocityVerletStep2,
            nblocks, nthreads, 0, stream,
            pvView, dt_2m);
    }

    // restore previous positions into old_particles channel
    pv->local()->dataPerParticle.getData<real4>(channel_names::oldPositions)->copy(previousPositions_, stream);

    invalidatePV_(pv);
}

void IntegratorSubStepShardlowSweep::setPrerequisites(ParticleVector *pv)
{
    if (auto *mv = dynamic_cast<MembraneVector*>(pv))
    {
        // luckily do not need cell lists for self interactions
        fastForces_->setPrerequisites(pv, pv, nullptr, nullptr);

        auto mesh = dynamic_cast<MembraneMesh*>(mv->mesh.get());
        assert(mesh);
        pvToEdgeSets_[mv->getName()] = std::make_unique<MeshDistinctEdgeSets>(mesh);
    }
    else
    {
        die("%s: expected a MembraneVector, got '%s'", getCName(), pv->getCName());
    }
}

void IntegratorSubStepShardlowSweep::_viscousSweeps(MembraneVector *mv, cudaStream_t stream)
{
    const real dt = getState()->getDt() / (substeps_ * nsweeps_);

    OVview ovview(mv, mv->local());

    const real invMass = 1.0_r / ovview.mass;
    const real sigma = math::sqrt(2 * gammaC_ * kBT_);

    const auto edgeSets = pvToEdgeSets_[mv->getName()].get();

    for (int sweep = 0; sweep < nsweeps_; ++sweep)
    {
        std::uniform_real_distribution<real> u(0.0_r, 1.0_r);

        for (int color = 0; color < edgeSets->numColors(); ++color)
        {
            const real seed = u(rnd_);
            const auto& edges = edgeSets->edgeSet(color);

            constexpr int nthreads = 128;
            const int nblocks = getNblocks(ovview.nObjects * edges.size(), nthreads);

            SAFE_KERNEL_LAUNCH(
                rbc_shardlow_kernels::sweepVelocities,
                nblocks, nthreads, 0, stream,
                edges.size(), edges.devPtr(), ovview, dt, gammaC_, sigma, invMass, seed);
        }
    }
}

} // namespace mirheo
