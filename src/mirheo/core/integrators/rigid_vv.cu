#include "rigid_vv.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/pvs/views/rov.h>
#include <mirheo/core/rigid/operations.h>
#include <mirheo/core/rigid/utils.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/quaternion.h>

namespace mirheo
{

namespace RigidVVKernels
{

__device__ static inline void performRotation(real dt, real3 J, real3 invJ, RigidMotion& motion)
{
    auto q = motion.q;

    // Update angular velocity in the body frame
    auto omega     = q.inverseRotate(motion.omega);
    const auto tau = q.inverseRotate(motion.torque);

    // tau = J dw/dt + w x Jw  =>  dw/dt = J_1*tau - J_1*(w x Jw)
    // J is the diagonal inertia tensor in the body frame
    const RigidReal3 dw_dt = invJ * (tau - cross(omega, J * omega));
    omega += dw_dt * dt;
    omega = q.rotate(omega);

    // using OLD q and NEW w ?
    // d^2q / dt^2 = 1/2 * (dw/dt*q + w*dq/dt)
    auto dq_dt = q.timeDerivative(omega);
    auto d2q_dt2 = 0.5 * (Quaternion<RigidReal>::pureVector(dw_dt) * q +
                          Quaternion<RigidReal>::pureVector(omega) * dq_dt);

    dq_dt += d2q_dt2 * dt;
    q     += dq_dt   * dt;

    // Normalize q
    q.normalize();

    motion.omega = omega;
    motion.q     = q;
}

__device__ static inline void performTranslation(real dt, real invMass, RigidMotion& motion)
{
    const auto force = motion.force;
    motion.vel += (dt * invMass) * force;
    motion.r   += dt * motion.vel;
}


/**
 * J is the diagonal moment of inertia tensor, J_1 is its inverse (simply 1/Jii)
 * Velocity-Verlet fused is used at the moment
 */
__global__ void integrateRigidMotion(ROVviewWithOldMotion view, real dt)
{
    const int objId = threadIdx.x + blockDim.x * blockIdx.x;
    if (objId >= view.nObjects) return;

    auto motion = view.motions[objId];
    view.old_motions[objId] = motion;

    performRotation   (dt, view.J, view.J_1, motion);
    performTranslation(dt, view.invObjMass,  motion);
    
    view.motions[objId] = motion;
}

} // namespace RigidVVKernels

IntegratorVVRigid::IntegratorVVRigid(const MirState *state, std::string name) :
    Integrator(state, name)
{}

IntegratorVVRigid::~IntegratorVVRigid() = default;

/**
 * Can only be applied to RigidObjectVector and requires it to have
 * \c old_motions data channel per particle
 */
void IntegratorVVRigid::setPrerequisites(ParticleVector *pv)
{
    auto ov = dynamic_cast<RigidObjectVector*> (pv);
    if (ov == nullptr)
        die("Rigid integration only works with rigid objects, can't work with %s", pv->name.c_str());

    ov->requireDataPerObject<RigidMotion>(ChannelNames::oldMotions, DataManager::PersistenceMode::None);
    warn("Only objects with diagonal inertia tensors are supported now for rigid integration");
}


// FIXME: split VV into two stages
void IntegratorVVRigid::stage1(__UNUSED ParticleVector *pv, __UNUSED cudaStream_t stream)
{}


static void integrateRigidMotions(const ROVviewWithOldMotion& view, real dt, cudaStream_t stream)
{
    const int nthreads = 64;
    const int nblocks = getNblocks(view.nObjects, nthreads);
    
    SAFE_KERNEL_LAUNCH(
        RigidVVKernels::integrateRigidMotion,
        nblocks, nthreads, 0, stream,
        view, dt );
}

void IntegratorVVRigid::stage2(ParticleVector *pv, cudaStream_t stream)
{
    const real dt = state->dt;
    auto rov = dynamic_cast<RigidObjectVector*> (pv);

    debug("Integrating %d rigid objects %s (total %d particles), timestep is %f",
          rov->local()->nObjects, rov->name.c_str(), rov->local()->size(), dt);

    const ROVviewWithOldMotion rovView(rov, rov->local());

    RigidOperations::collectRigidForces(rovView, stream);

    integrateRigidMotions(rovView, dt, stream);

    RigidOperations::applyRigidMotion(rovView, rov->initialPositions,
                                      RigidOperations::ApplyTo::PositionsAndVelocities, stream);

    invalidatePV(pv);
}

} // namespace mirheo
