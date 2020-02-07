#include <mirheo/core/datatypes.h>
#include <mirheo/core/integrators/rigid_vv.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/rigid_object_vector.h>
#include <mirheo/core/rigid/utils.h>
#include <mirheo/core/utils/common.h>
#include <mirheo/core/utils/cuda_common.h>

#include <cstdio>
#include <gtest/gtest.h>

using namespace mirheo;

static inline RigidMotion initMotion(real3 omega)
{
    constexpr RigidReal3 zero3 {0.0_r, 0.0_r, 0.0_r};
    
    RigidMotion m;
    m.r = m.vel = m.force = m.torque = zero3;
    m.q = Quaternion<RigidReal>::createFromComponents(1.0, 0.0, 0.0, 0.0);
    m.omega = make_rigidReal3(omega);
    return m;
}

static inline std::vector<real3> makeTemplate(real x = 1.0_r, real y = 1.0_r, real z = 1.0_r)
{
    return {{+x, 0.0_r, 0.0_r},
            {-x, 0.0_r, 0.0_r},
            {0.0_r, +y, 0.0_r},
            {0.0_r, -y, 0.0_r},
            {0.0_r, 0.0_r, +z},
            {0.0_r, 0.0_r, -z}};
}

static PinnedBuffer<real4> getInitialPositions(const std::vector<real3>& in, cudaStream_t stream)
{
    PinnedBuffer<real4> out(in.size());
    
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = make_real4(in[i].x, in[i].y, in[i].z, 0);
        
    out.uploadToDevice(stream);
    return out;
}

static void setParticlesFromMotions(RigidObjectVector *rov, cudaStream_t stream)
{
    // use rigid object integrator to set up the particles positions, velocities and old positions
    rov->local()->forces().clear(stream);
    const real dummyDt = 0._r;
    const MirState dummyState(rov->getState()->domain, dummyDt);
    IntegratorVVRigid integrator(&dummyState, "__dummy__");
    integrator.execute(rov, stream);
}

static inline std::unique_ptr<RigidObjectVector> makeRigidVector(const MirState *state, real mass, real3 J, real3 omega, cudaStream_t stream)
{
    const auto posTemplate = makeTemplate();
    const int objSize = posTemplate.size();
    const int nObjects = 1;
    const real partMass = mass / objSize;

    auto rov = std::make_unique<RigidObjectVector>(state, "rigid_body", partMass, J, objSize, std::make_shared<Mesh>(), nObjects);

    auto lrov = rov->local();
    auto& motions = *lrov->dataPerObject.getData<RigidMotion>(ChannelNames::motions);
    
    for (auto& m : motions)
        m = initMotion(omega);
    motions.uploadToDevice(stream);

    rov->initialPositions = getInitialPositions(posTemplate, stream);
    setParticlesFromMotions(rov.get(), stream);
    
    return rov;
}

struct Params
{
    real dt   {1e-3_r};
    real tend {10.0_r};
    real mass {1.0_r};
    real3 J, omega;
    real3 torque {0.0_r, 0.0_r, 0.0_r};
};

static inline RigidMotion advanceGPU(const Params& p)
{
    constexpr real L = 32.0_r;
    
    DomainInfo domain {{L, L, L}, {0._r, 0._r, 0._r}, {L, L, L}};
    MirState state(domain, p.dt);

    IntegratorVVRigid gpuIntegrator(&state, "rigid_vv");
    auto rov = makeRigidVector(&state, p.mass, p.J, p.omega, defaultStream);

    gpuIntegrator.setPrerequisites(rov.get());

    for (; state.currentTime < p.tend; state.currentTime += state.dt)
    {
        gpuIntegrator.execute(rov.get(), defaultStream);
    }

    auto& motions = *rov->local()->dataPerObject.getData<RigidMotion>(ChannelNames::motions);
    motions.downloadFromDevice(defaultStream);
    
    return motions[0];
}

enum class RotationScheme {Original, ConsistentQ};

static inline void advanceRotation(real dt, real3 J, real3 invJ, RigidMotion& motion)
{
    auto q = motion.q;

    auto omega = q.inverseRotate(motion.omega);
    auto tau   = q.inverseRotate(motion.torque);

    const RigidReal3 dw_dt = invJ * (tau - cross(omega, J * omega));
    omega += dw_dt * dt;
    omega = q.rotate(omega);

    auto dq_dt = q.timeDerivative(omega);
    const auto d2q_dt2 = 0.5 * (Quaternion<RigidReal>::pureVector(dw_dt) * q +
                                Quaternion<RigidReal>::pureVector(omega) * dq_dt);

    dq_dt += d2q_dt2 * dt;
    q     += dq_dt   * dt;

    motion.omega = omega;
    motion.q     = q.normalized();
}

static inline void advanceRotationConsistentQ(real dt, real3 J, real3 invJ, RigidMotion& motion)
{
    constexpr RigidReal tol = 1e-12;
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
    auto qhalf = (q + dt_half * dqhalf_dt).normalized();

    RigidReal err = tol + 1.0; // to make sure we are above the tolerance
    for (int iter  = 0; iter < maxIter && err > tol; ++iter)
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

static inline void advanceTranslation(real dt, real invMass, RigidMotion& motion)
{
    const auto force = motion.force;
    auto vel   = motion.vel;
    vel += force*dt * invMass;

    motion.vel = vel;
    motion.r += vel*dt;
}

template <RotationScheme rotScheme>
static inline void advanceOneStep(real dt, real3 J, real3 invJ, real invMass, RigidMotion& motion)
{
    if (rotScheme == RotationScheme::ConsistentQ)
        advanceRotationConsistentQ(dt, J, invJ, motion);
    else
        advanceRotation(dt, J, invJ, motion);
    
    advanceTranslation(dt, invMass, motion);
}

template <RotationScheme rotScheme>
static inline RigidMotion advanceCPU(const Params& p, RigidMotion m)
{
    using TimeType = MirState::TimeType;
    const TimeType dt = p.dt;

    const real3 invJ = 1.0_r / p.J;
    const real invMass = 1.0 / p.mass;
    
    for (TimeType t = 0; t < p.tend; t += dt)
        advanceOneStep<rotScheme>(dt, p.J, invJ, invMass, m);

    return m;
}

template <RotationScheme rotScheme>
static inline RigidMotion advanceCPU(const Params& p)
{
    auto m = initMotion(p.omega);
    m.torque = make_rigidReal3(p.torque);
    return advanceCPU<rotScheme>(p, m);
}


TEST (Integration_Rigid, GPU_CPU_compare)
{
    Params p;
    p.J     = make_real3(1.0_r, 2.0_r, 3.0_r);
    p.omega = make_real3(10.0_r, 5.0_r, 4.0_r);

    const auto gpuM = advanceGPU(p);
    const auto cpuM = advanceCPU<RotationScheme::ConsistentQ>(p);

    constexpr real tol = 1e-6;
    ASSERT_NEAR(gpuM.q.w, cpuM.q.w, tol);
    ASSERT_NEAR(gpuM.q.x, cpuM.q.x, tol);
    ASSERT_NEAR(gpuM.q.y, cpuM.q.y, tol);
    ASSERT_NEAR(gpuM.q.z, cpuM.q.z, tol);

    ASSERT_NEAR(gpuM.omega.x, cpuM.omega.x, tol);
    ASSERT_NEAR(gpuM.omega.y, cpuM.omega.y, tol);
    ASSERT_NEAR(gpuM.omega.z, cpuM.omega.z, tol);
}


TEST (Integration_Rigid, Analytic_CPU_compare_principal_axes)
{
    auto check = [](real3 omega)
    {
        Params p;
        p.J     = make_real3(1.0_r, 2.0_r, 3.0_r);
        p.omega = omega;

        const auto cpuM = advanceCPU<RotationScheme::ConsistentQ>(p);
        
        constexpr real tol = 1e-6;
        ASSERT_NEAR(omega.x, cpuM.omega.x, tol);
        ASSERT_NEAR(omega.y, cpuM.omega.y, tol);
        ASSERT_NEAR(omega.z, cpuM.omega.z, tol);
    };

    check({12.0_r, 0.0_r, 0.0_r});
    check({0.0_r, 10.0_r, 0.0_r});
    check({0.0_r, 0.0_r, -2.0_r});
}

static inline real3 computeAngularMomentum(real3 J, const RigidMotion& m)
{
    const auto omega = m.q.inverseRotate(m.omega);
    const auto L = omega * make_rigidReal3(J);
    return make_real3(m.q.rotate(L));
}

TEST (Integration_Rigid, L_is_conserved)
{
    constexpr auto rotScheme = RotationScheme::ConsistentQ;
    Params p;
    p.J     = make_real3(20.0_r, 30.0_r, 10.0_r);
    p.omega = make_real3(-2.0_r, 5.0_r, -1.4_r);

    p.tend = 1.0_r;
    p.dt = 1e-5_r;
    const int nsteps = 10;
    RigidMotion motion = advanceCPU<rotScheme>(p);

    real3 Lprev = computeAngularMomentum(p.J, motion);

    for (int i = 0; i < nsteps; ++i)
    {
        motion = advanceCPU<rotScheme>(p, motion);
        const real3 L = computeAngularMomentum(p.J, motion);

        const real err = length(L - Lprev) / length(L);
        ASSERT_LE(err, 1e-4_r);
        // printf("%g %g %g %g\n", err, L.x, L.y, L.z);
        Lprev = L;
    }
}

TEST (Integration_Rigid, constantTorquePrincipalAxis)
{
    // assume two components are 0
    auto check = [](real3 torque)
    {
        Params p;
        p.J     = make_real3(5.0_r, 2.0_r, 3.0_r);
        p.omega = make_real3(0.0_r, 0.0_r, 0.0_r);
        p.torque = torque;

        const auto cpuM = advanceCPU<RotationScheme::ConsistentQ>(p);

        const real3 invJ = 1.0_r / p.J;
        const auto omegaRef = p.tend * invJ * torque;
        
        const real3 tol = math::abs(1e-6_r * omegaRef);
        ASSERT_NEAR(omegaRef.x, cpuM.omega.x, tol.x);
        ASSERT_NEAR(omegaRef.y, cpuM.omega.y, tol.y);
        ASSERT_NEAR(omegaRef.z, cpuM.omega.z, tol.z);
    };

    check({12.0_r, 0.0_r, 0.0_r});
    check({0.0_r, 10.0_r, 0.0_r});
    check({0.0_r, 0.0_r, -2.0_r});
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
