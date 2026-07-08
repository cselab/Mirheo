#include <mirheo/core/logger.h>
#include <mirheo/core/analytical_shapes/api.h>

#include <cstdio>
#include <gtest/gtest.h>
#include <random>

using namespace mirheo;

template<class Shape>
static real3 inertiaTensorMC(long nsamples, const Shape& shape, real3 low, real3 high)
{
    double V, xx, yy, zz, xy, xz, yz;
    V = xx = xy = xz = yy = yz = zz = 0;

    const long seed = 424242424242;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<real> distx(low.x, high.x);
    std::uniform_real_distribution<real> disty(low.y, high.y);
    std::uniform_real_distribution<real> distz(low.z, high.z);

    for (long i = 0; i < nsamples; ++i)
    {
        real3 r {distx(gen), disty(gen), distz(gen)};

        if (shape.inOutFunction(r) < 0.f)
        {
            V += 1.0;
            xx += r.x * r.x;
            xy += r.x * r.y;
            xz += r.x * r.z;
            yy += r.y * r.y;
            yz += r.y * r.z;
            zz += r.z * r.z;
        }
    }

    if (V > 0)
    {
        xx /= V;
        yy /= V;
        zz /= V;
    }

    auto B = high - low;
    printf("%g\n", V / nsamples);
    printf("V = %g\n", V * B.x * B.y * B.z / nsamples);
    printf("%g %g %g\n", xy / nsamples, xz / nsamples, yz / nsamples);

    real3 I {real(yy + zz),
              real(xx + zz),
              real(xx + yy)};
    return I;
}

static real Lmax(real3 a, real3 b)
{
    return math::max(math::max(math::abs(a.x-b.x), math::abs(a.y-b.y)), math::abs(a.z-b.z));
}

TEST (InertiaTensor, Ellipsoid)
{
    real3 axes {1._r, 2._r, 3._r};
    Ellipsoid ell(axes);

    real3 Iref = inertiaTensorMC(1000000, ell, -axes, axes);
    real3 I    = ell.inertiaTensor(1.0);

    // printf("%g %g %g   %g %g %g\n",
    //        Iref.x, Iref.y, Iref.z,
    //        I.x, I.y, I.z);

    ASSERT_LE(Lmax(I, Iref), 1e-2);
}

TEST (InertiaTensor, Cylinder)
{
    real L = 5.0_r;
    real R = 3.0_r;
    real3 lim {R, R, 0.55_r * L};
    Cylinder cyl(R, L);

    real3 Iref = inertiaTensorMC(1000000, cyl, -lim, lim);
    real3 I    = cyl.inertiaTensor(1.0);

    // printf("%g %g %g   %g %g %g\n",
    //        Iref.x, Iref.y, Iref.z,
    //        I.x, I.y, I.z);

    ASSERT_LE(Lmax(I, Iref), 1e-2);
}

TEST (InertiaTensor, Capsule)
{
    real L = 5.0_r;
    real R = 3.0_r;
    real3 lim {R, R, 0.55_r * L + R};
    Capsule cap(R, L);

    real3 Iref = inertiaTensorMC(10000000, cap, -lim, lim);
    real3 I    = cap.inertiaTensor(1.0);

    printf("%g %g %g   %g %g %g\n",
           Iref.x, Iref.y, Iref.z,
           I.x, I.y, I.z);

    ASSERT_LE(Lmax(I, Iref), 1e-2);
}

int main(int argc, char **argv)
{
    logger.init(MPI_COMM_NULL, "inertia_tensor.log", 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
