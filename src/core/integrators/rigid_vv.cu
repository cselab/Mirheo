#include "rigid_vv.h"

#include <core/logger.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/views/rov.h>
#include <core/rigid/operations.h>
#include <core/rigid/utils.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/quaternion.h>

namespace RigidVVKernels
{

/**
 * J is the diagonal moment of inertia tensor, J_1 is its inverse (simply 1/Jii)
 * Velocity-Verlet fused is used at the moment
 */
__global__ void integrateRigidMotion(ROVviewWithOldMotion ovView, const float dt)
{
    const int objId = threadIdx.x + blockDim.x * blockIdx.x;
    if (objId >= ovView.nObjects) return;

    auto motion = ovView.motions[objId];
    ovView.old_motions[objId] = motion;

    //**********************************************************************************
    // Rotation
    //**********************************************************************************
    auto q = motion.q;

    // Update angular velocity in the body frame
    auto omega = Quaternion::rotate(motion.omega,  Quaternion::invQ(q));
    auto tau   = Quaternion::rotate(motion.torque, Quaternion::invQ(q));

    // tau = J dw/dt + w x Jw  =>  dw/dt = J_1*tau - J_1*(w x Jw)
    // J is the diagonal inertia tensor in the body frame
    auto dw_dt = ovView.J_1 * (tau - cross(omega, ovView.J*omega));
    omega += dw_dt * dt;

    // Only for output purposes
    auto L = Quaternion::rotate(omega*ovView.J, motion.q);

    omega = Quaternion::rotate(omega, motion.q);

    // using OLD q and NEW w ?
    // d^2q / dt^2 = 1/2 * (dw/dt*q + w*dq/dt)
    auto dq_dt = Quaternion::compute_dq_dt(q, omega);
    auto d2q_dt2 = 0.5f*(Quaternion::multiply(Quaternion::f3toQ(dw_dt), q) +
                         Quaternion::multiply(Quaternion::f3toQ(omega), dq_dt));

    dq_dt += d2q_dt2 * dt;
    q     += dq_dt   * dt;

    // Normalize q
    q = normalize(q);

    motion.omega = omega;
    motion.q     = q;

    //**********************************************************************************
    // Translation
    //**********************************************************************************
    auto force = motion.force;
    auto vel   = motion.vel;
    vel += force*dt * ovView.invObjMass;

    motion.vel = vel;
    motion.r += vel*dt;

    ovView.motions[objId] = motion;
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


static void integrateRigidMotions(const ROVviewWithOldMotion& view, float dt, cudaStream_t stream)
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
    const float dt = state->dt;
    auto rov = dynamic_cast<RigidObjectVector*> (pv);

    debug("Integrating %d rigid objects %s (total %d particles), timestep is %f",
          rov->local()->nObjects, rov->name.c_str(), rov->local()->size(), dt);

    const ROVviewWithOldMotion rovView(rov, rov->local());

    RigidOperations::collectRigidForces(rovView, stream);

    integrateRigidMotions(rovView, dt, stream);

    RigidOperations::applyRigidMotion(rovView, rov->initialPositions,
                                      RigidOperations::ApplyTo::PositionsAndVelocities, stream);

    RigidOperations::clearRigidForces(rovView, stream);

    invalidatePV(pv);
}

