#include <core/logger.h>
#include <core/analytical_shapes/api.h>

#include <cstdio>
#include <gtest/gtest.h>
#include <random>

Logger logger;

template<class Shape>
static float3 inertiaTensorMC(long nsamples, const Shape& shape, float3 low, float3 high)
{
    double V, xx, yy, zz, xy, xz, yz;
    V = xx = xy = xz = yy = yz = zz = 0;

    const long seed = 424242424242;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> distx(low.x, high.x);
    std::uniform_real_distribution<float> disty(low.y, high.y);
    std::uniform_real_distribution<float> distz(low.z, high.z);
    
    for (long i = 0; i < nsamples; ++i)
    {
        float3 r {distx(gen), disty(gen), distz(gen)};

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
    
    float3 I {float(yy + zz),
              float(xx + zz),
              float(xx + yy)};
    return I;
}

static float Lmax(float3 a, float3 b)
{
    return max(max(fabs(a.x-b.x), fabs(a.y-b.y)), fabs(a.z-b.z));
}

TEST (InertiaTensor, Ellipsoid)
{
    float3 axes {1.f, 2.f, 3.f};
    Ellipsoid ell(axes);

    float3 Iref = inertiaTensorMC(1000000, ell, -axes, axes);
    float3 I    = ell.inertiaTensor(1.0);

    // printf("%g %g %g   %g %g %g\n",
    //        Iref.x, Iref.y, Iref.z,
    //        I.x, I.y, I.z);
    
    ASSERT_LE(Lmax(I, Iref), 1e-2);
}

TEST (InertiaTensor, Cylinder)
{
    float L = 5.0;
    float R = 3.0;
    float3 lim {R, R, 0.55f * L};
    Cylinder cyl(R, L);

    float3 Iref = inertiaTensorMC(1000000, cyl, -lim, lim);
    float3 I    = cyl.inertiaTensor(1.0);

    // printf("%g %g %g   %g %g %g\n",
    //        Iref.x, Iref.y, Iref.z,
    //        I.x, I.y, I.z);

    ASSERT_LE(Lmax(I, Iref), 1e-2);
}

TEST (InertiaTensor, Capsule)
{
    float L = 5.0;
    float R = 3.0;
    float3 lim {R, R, 0.55f * L + R};
    Capsule cap(R, L);

    float3 Iref = inertiaTensorMC(10000000, cap, -lim, lim);
    float3 I    = cap.inertiaTensor(1.0);

    printf("%g %g %g   %g %g %g\n",
           Iref.x, Iref.y, Iref.z,
           I.x, I.y, I.z);

    ASSERT_LE(Lmax(I, Iref), 1e-2);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
