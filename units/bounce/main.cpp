#include <mirheo/core/logger.h>
#include <mirheo/core/bouncers/kernels/api.h>

#include <gtest/gtest.h>

using namespace mirheo;

TEST (BounceKernels, BounceMaxwell_returns_correct_velocity_orientation)
{
    const real kBT = 2.0_r;
    const real mass = 0.5_r;

    const real3 nWall = normalize(make_real3(1.0_r, -2.0_r, 3.0_r));
    const real3 uWall = make_real3(0.0_r, -1.0_r, 0.0_r);
    const real3 uOld = make_real3(1.0_r, -2.0_r, 3.0_r);

    std::mt19937 gen(0XC0FFEE);
    BounceMaxwell bouncer{kBT};

    for (int i = 0; i < 200; ++i)
    {
        bouncer.update(gen);
        const real3 uNew = bouncer.newVelocity(uOld, uWall, nWall, mass);
        ASSERT_GE(dot(uNew-uWall, nWall), 0.0_r);
    }
}

TEST (BounceKernels, BounceBack_creates_correct_velocity)
{
    const real mass = 0.5_r;

    const real3 nWall = normalize(make_real3(1.0_r, -2.0_r, 3.0_r));
    const real3 uWall = make_real3(0.0_r, -1.0_r, 0.0_r);
    const real3 uOld = make_real3(1.0_r, -2.0_r, 3.0_r);

    std::mt19937 gen(0XC0FFEE);
    BounceBack bouncer;

    for (int i = 0; i < 200; ++i)
    {
        bouncer.update(gen);
        const real3 uNew = bouncer.newVelocity(uOld, uWall, nWall, mass);

        const real3 du = 0.5_r * (uNew + uOld) - uWall;
        constexpr real tol = 1e-6_r;
        ASSERT_NEAR(du.x, 0.0_r, tol);
        ASSERT_NEAR(du.y, 0.0_r, tol);
        ASSERT_NEAR(du.z, 0.0_r, tol);
    }
}





int main(int argc, char **argv)
{
    logger.init(MPI_COMM_NULL, "bounce.log", 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
