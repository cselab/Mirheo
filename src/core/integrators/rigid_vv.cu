#include "rigid_vv.h"

#include <core/utils/kernel_launch.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>


#include <core/rigid_kernels/integration.h>

IntegratorVVRigid::IntegratorVVRigid(const MirState *state, std::string name) :
    Integrator(state, name)
{}

IntegratorVVRigid::~IntegratorVVRigid() = default;

/**
 * Can only be applied to RigidObjectVector and requires it to have
 * \c old_motions data channel per particle
 */
void IntegratorVVRigid::setPrerequisites(ParticleVector* pv)
{
    auto ov = dynamic_cast<RigidObjectVector*> (pv);
    if (ov == nullptr)
        die("Rigid integration only works with rigid objects, can't work with %s", pv->name.c_str());

    ov->requireDataPerObject<RigidMotion>(ChannelNames::oldMotions, DataManager::PersistenceMode::None);
    warn("Only objects with diagonal inertia tensors are supported now for rigid integration");
}


// FIXME: split VV into two stages
void IntegratorVVRigid::stage1(ParticleVector *pv, cudaStream_t stream)
{}




static void collectRigidForces(const ROVviewWithOldMotion& view, cudaStream_t stream)
{
    const int nthreads = 128;
    const int nblocks = getNblocks(2*view.size, nthreads);
    
    SAFE_KERNEL_LAUNCH(
        RigidIntegrationKernels::collectRigidForces,
        nblocks, nthreads, 0, stream,
        view );
}

static void integrateRigidMotions(const ROVviewWithOldMotion& view, float dt, cudaStream_t stream)
{
    const int nthreads = 64;
    const int nblocks = getNblocks(view.nObjects, nthreads);
    
    SAFE_KERNEL_LAUNCH(
        RigidIntegrationKernels::integrateRigidMotion,
        nblocks, nthreads, 0, stream,
        view, dt );
}

static void setRigidParticlesFromMotions(const ROVviewWithOldMotion& view, const PinnedBuffer<float4>& initialPositions, cudaStream_t stream)
{
    const int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);
        
    SAFE_KERNEL_LAUNCH(
        RigidIntegrationKernels::applyRigidMotion<RigidIntegrationKernels::ApplyRigidMotion::PositionsAndVelocities>,
        nblocks, nthreads, 0, stream,
        view, initialPositions.devPtr() );
}

static void clearRigidForces(const ROVviewWithOldMotion& view, cudaStream_t stream)
{
    const int nthreads = 64;
    const int nblocks = getNblocks(view.nObjects, nthreads);

    SAFE_KERNEL_LAUNCH(
        RigidIntegrationKernels::clearRigidForces,
        nblocks, nthreads, 0, stream,
        view );
}


void IntegratorVVRigid::stage2(ParticleVector *pv, cudaStream_t stream)
{
    const float dt = state->dt;
    auto rov = dynamic_cast<RigidObjectVector*> (pv);

    debug("Integrating %d rigid objects %s (total %d particles), timestep is %f",
          rov->local()->nObjects, rov->name.c_str(), rov->local()->size(), dt);

    const ROVviewWithOldMotion rovView(rov, rov->local());

    collectRigidForces           (rovView,                        stream);
    integrateRigidMotions        (rovView, dt,                    stream);    
    setRigidParticlesFromMotions (rovView, rov->initialPositions, stream);
    clearRigidForces             (rovView,                        stream);

    invalidatePV(pv);
}

