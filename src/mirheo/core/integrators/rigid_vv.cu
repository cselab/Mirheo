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

// http://lab.pdebuyl.be/rmpcdmd/algorithms/quaternions.html
__device__ static inline void performRotation(real dt, real3 J, real3 invJ, RigidMotion& motion)
{
    constexpr RigidReal tol = 1e-10;
    constexpr int maxIter = 50;

    const RigidReal dt_half = 0.5 * dt;
    auto q = motion.q;

    const RigidReal3 omegaB  = q.inverseRotate(motion.omega);
    const RigidReal3 torqueB = q.inverseRotate(motion.torque);

    const RigidReal3 LB0   = omegaB * make_rigidReal3(J);
    const RigidReal3 L0    = q.rotate(LB0);
    const RigidReal3 Lhalf = L0 + dt_half * motion.torque;

    const RigidReal3 dLB0_dt = torqueB - cross(omegaB, LB0);
    RigidReal3 LBhalf     = LB0 + dt_half * dLB0_dt;
    RigidReal3 omegaBhalf = make_rigidReal3(invJ) * LBhalf;

    // iteration: find consistent dqhalf_dt such that it is self consistent
    auto dqhalf_dt = 0.5 * q * RigiQuaternion::pureVector(omegaBhalf);
    auto qhalf     = (q + dt_half * dqhalf_dt).normalized();

    RigidReal err = tol + 1.0; // to make sure we are above the tolerance
    for (int iter = 0; iter < maxIter && err > tol; ++iter)
    {
        const auto qhalf_prev = qhalf;
        LBhalf     = qhalf.inverseRotate(Lhalf);
        omegaBhalf = make_rigidReal3(invJ) * LBhalf;
        dqhalf_dt  = 0.5 * qhalf * RigiQuaternion::pureVector(omegaBhalf);
        qhalf      = (q + dt_half * dqhalf_dt).normalized();

        err = (qhalf - qhalf_prev).norm();
    }

    q += dt * dqhalf_dt;
    q.normalize();

    const RigidReal3 dw_dt = invJ * (torqueB - cross(omegaB, J * omegaB));
    const RigidReal3 omegaB1 = omegaB + dw_dt * dt;
    motion.omega = q.rotate(omegaB1);
    motion.q = q;
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

IntegratorVVRigid::IntegratorVVRigid(const MirState *state, const std::string& name) :
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
        die("Rigid integration only works with rigid objects, can't work with %s", pv->getCName());

    ov->requireDataPerObject<RigidMotion>(ChannelNames::oldMotions, DataManager::PersistenceMode::None);
    // warn("Only objects with diagonal inertia tensors are supported now for rigid integration");
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
    const real dt = getState()->dt;
    auto rov = dynamic_cast<RigidObjectVector*> (pv);

    debug("Integrating %d rigid objects %s (total %d particles), timestep is %f",
          rov->local()->nObjects, rov->getCName(), rov->local()->size(), dt);

    const ROVviewWithOldMotion rovView(rov, rov->local());

    RigidOperations::collectRigidForces(rovView, stream);

    integrateRigidMotions(rovView, dt, stream);

    RigidOperations::applyRigidMotion(rovView, rov->initialPositions,
                                      RigidOperations::ApplyTo::PositionsAndVelocities, stream);

    invalidatePV_(pv);
}

} // namespace mirheo
