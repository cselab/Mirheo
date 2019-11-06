#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/rigid/rigid_motion.h>
#include <mirheo/core/utils/quaternion.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

using namespace mirheo;

namespace mirheo { Logger logger; }

inline void stage1(RigidMotion& motion, float dt, float3 J, float3 Jinv)
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

inline void stage2(RigidMotion& motion, float dt, float3 J, float3 Jinv)
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
