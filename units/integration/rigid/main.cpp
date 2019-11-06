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

namespace mirheo { Logger logger; }

static inline RigidMotion initMotion(real3 omega)
{
    constexpr RigidReal3 zero3 {0.0_r, 0.0_r, 0.0_r};
    
    RigidMotion m;
    m.r = m.vel = m.force = m.torque = zero3;
    m.q = Quaternion<RigidReal>::createFromComponents(1.0, 0.0, 0.0, 0.0);
    m.omega = make_rigidReal3(omega);
    return m;
}

static inline std::vector<float3> makeTemplate(real x = 1.0_r, real y = 1.0_r, real z = 1.0_r)
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
    const MirState dummyState(rov->state->domain, dummyDt);
    IntegratorVVRigid integrator(&dummyState, "__dummy__");
    integrator.stage2(rov, stream);
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
        gpuIntegrator.stage1(rov.get(), defaultStream);
        gpuIntegrator.stage2(rov.get(), defaultStream);
    }

    auto& motions = *rov->local()->dataPerObject.getData<RigidMotion>(ChannelNames::motions);
    motions.downloadFromDevice(defaultStream);
    
    return motions[0];
}


static inline void advanceOneStep(real dt, real3 J, real3 invJ, real invMass, RigidMotion& motion)
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

    const auto force = motion.force;
    auto vel   = motion.vel;
    vel += force*dt * invMass;

    motion.vel = vel;
    motion.r += vel*dt;
}

static inline RigidMotion advanceCPU(const Params& p)
{
    using TimeType = MirState::TimeType;
    const TimeType dt = p.dt;

    auto m = initMotion(p.omega);

    const real3 invJ = 1.0_r / p.J;
    const real invMass = 1.0 / p.mass;
    
    for (TimeType t = 0; t < p.tend; t += dt)
        advanceOneStep(dt, p.J, invJ, invMass, m);

    return m;
}

TEST (RIGID_MOTION, GPU_CPU_compare)
{
    Params p;
    p.J     = make_real3(1.0_r, 2.0_r, 3.0_r);
    p.omega = make_real3(10.0_r, 5.0_r, 4.0_r);

    const auto gpuM = advanceGPU(p);
    const auto cpuM = advanceCPU(p);

    constexpr real tol = 1e-6;
    ASSERT_NEAR(gpuM.q.w, cpuM.q.w, tol);
    ASSERT_NEAR(gpuM.q.x, cpuM.q.x, tol);
    ASSERT_NEAR(gpuM.q.y, cpuM.q.y, tol);
    ASSERT_NEAR(gpuM.q.z, cpuM.q.z, tol);

    ASSERT_NEAR(gpuM.omega.x, cpuM.omega.x, tol);
    ASSERT_NEAR(gpuM.omega.y, cpuM.omega.y, tol);
    ASSERT_NEAR(gpuM.omega.z, cpuM.omega.z, tol);
}




static inline void stage1(RigidMotion& motion, float dt, float3 J, float3 Jinv)
{
    // http://lab.pdebuyl.be/rmpcdmd/algorithms/quaternions.html
    const double dt_half = 0.5 * dt;

    const auto q0 = motion.q;
    const auto invq0 = q0.conjugate();
    const auto omegaB  = invq0.rotate(motion.omega);
    const auto torqueB = invq0.rotate(motion.torque);
    const auto LB = J * omegaB;
    const auto L0 = q0.rotate(LB);

    const auto L_half = L0 + dt_half * motion.torque;

    const auto dLB0_dt = torqueB - cross(omegaB, LB);

    
    constexpr RigidReal tolerance = 1e-6;

    auto LB_half     = LB + dt_half * dLB0_dt;
    auto omegaB_half = Jinv * LB_half;

    auto dq_dt_half = q0.timeDerivative(omegaB_half);
    auto q_half     = (q0 + dt_half * dq_dt_half).normalized();

    auto performIteration = [&]()
    {
        LB_half     = q_half.inverseRotate(L_half);
        omegaB_half = Jinv * LB_half;

        dq_dt_half = q_half.timeDerivative(omegaB_half);
        q_half     = (q0 + dt_half * dq_dt_half).normalized();
    };

    performIteration();
    auto q_half_prev = q_half;
    RigidReal err = 1.0 + tolerance;

    while (err > tolerance)
    {
        performIteration();
        err = (q_half - q_half_prev).norm();
        q_half_prev = q_half;
    }

    motion.q = (q0 + dt * dq_dt_half).normalized();
    motion.omega = motion.q.rotate(omegaB_half);
}

static inline void stage2(RigidMotion& motion, float dt, float3 J, float3 Jinv)
{
    const double dt_half = 0.5 * dt;

    const auto q = motion.q;
    auto omegaB  = q.inverseRotate(motion.omega);
    auto LB = J * omegaB;
    auto L  = q.rotate(motion.omega);
    L += dt_half * motion.torque;
    LB = q.inverseRotate(L);
    omegaB = Jinv * LB;
    motion.omega = q.rotate(omegaB);
}

TEST (Integration_rigids, Analytic)
{
    // TODO
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
