#include <mirheo/core/logger.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/quaternion.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

Logger logger;

inline void stage1(RigidMotion& motion, float dt, float3 J, float3 Jinv)
{
    // http://lab.pdebuyl.be/rmpcdmd/algorithms/quaternions.html
    const double dt_half = 0.5 * dt;

    const auto q0 = motion.q;
    const auto invq0 = Quaternion::conjugate(q0);
    const auto omegaB  = Quaternion::rotate(motion.omega,  invq0);
    const auto torqueB = Quaternion::rotate(motion.torque, invq0);
    const auto LB = J * omegaB;
    const auto L0 = Quaternion::rotate(LB, q0);

    const auto L_half = L0 + dt_half * motion.torque;

    const auto dLB0_dt = torqueB - cross(omegaB, LB);

    
    constexpr RigidReal tolerance = 1e-6;

    auto LB_half     = LB + dt_half * dLB0_dt;
    auto omegaB_half = Jinv * LB_half;

    auto dq_dt_half = Quaternion::timeDerivative(q0, omegaB_half);
    auto q_half     = normalize(q0 + dt_half * dq_dt_half);

    auto performIteration = [&]()
    {
        LB_half     = Quaternion::rotate(L_half, Quaternion::conjugate(q_half));
        omegaB_half = Jinv * LB_half;

        dq_dt_half = Quaternion::timeDerivative(q_half, omegaB_half);
        q_half     = normalize(q0 + dt_half * dq_dt_half);
    };

    performIteration();
    auto q_half_prev = q_half;
    RigidReal err = 1.0 + tolerance;

    while (err > tolerance)
    {
        performIteration();
        err = length(q_half - q_half_prev);
        q_half_prev = q_half;
    }

    motion.q = normalize(q0 + dt * dq_dt_half);
    motion.omega = Quaternion::rotate(omegaB_half, motion.q);
}

inline void stage2(RigidMotion& motion, float dt, float3 J, float3 Jinv)
{
    const double dt_half = 0.5 * dt;

    const auto q = motion.q;
    const auto invq = Quaternion::conjugate(q);
    auto omegaB  = Quaternion::rotate(motion.omega,  invq);
    auto LB = J * omegaB;
    auto L  = Quaternion::rotate(motion.omega, q);
    L += dt_half * motion.torque;
    LB = Quaternion::rotate(L, invq);
    omegaB = Jinv * LB;
    motion.omega = Quaternion::rotate(omegaB, q);
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
