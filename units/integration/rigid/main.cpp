#include <core/logger.h>
#include <core/datatypes.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

Logger logger;

// void advance(RigidMotion& motion, float dt, float3 J, float3 Jinv)
// {
//     // http://lab.pdebuyl.be/rmpcdmd/algorithms/quaternions.html
//     auto q0 = motion.q;
//     auto omegaB  = rotate(motion.omega,  invQ(q));
//     auto torqueB = rotate(motion.torque, invQ(q));

//     auto domega_B_dt = J_inv * torqueB - cross(J_inv * omegaB, J * omegaB);
//     auto omegaB_half = omegaB + 0.5 * dt * domega_B_dt;


//     // initialize iterations
//     auto dq_dt_half = 0.5 * multiplyQ(q0, {RigidReal(0.0), omegaB_half.x, omegaB_half.y, omegaB_half.z});
//     auto q_half     = q0 + 0.5 * dt * dq_dt_half;

//     // iterate
//     while ()
//     {
//         omegaB_half = rotate
//     };
    
// }

TEST (Integration_rigids, Analytic)
{
    // TODO
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
