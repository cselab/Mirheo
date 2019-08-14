#include <core/logger.h>
#include <core/utils/helper_math.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>

Logger logger;

using real = double;
using real3 = double3;

struct Triangle
{
    real3 a, b, c;
};

static Triangle genTriangle(long seed)
{    
    Triangle t;
    srand48(seed);

    auto gen3 = []() -> real3 { return {(real) drand48(), (real) drand48(), (real) drand48()}; };
    
    t.a = gen3();
    t.b = gen3();
    t.c = gen3();

    return t;
};

static void shuffle(Triangle& t)
{
    auto tmp = t.a;
    t.a = t.b;
    t.b = t.c;
    t.c = tmp;
}

static real area(Triangle t)
{
    real3 v12 = t.b - t.a;
    real3 v13 = t.c - t.a;
    real3 n = cross(v12, v13);
    return 0.5 * length(n);
}

static real alpha(Triangle t, Triangle tref)
{
    return area(t) / area(tref) - 1;
}

inline real safeSqrt(real a)
{
    return a > 0.0 ? sqrt(a) : 0.0;
}

static real beta_stable(Triangle t, Triangle tref)
{
    real3 v12 = t.b - t.a;
    real3 v13 = t.c - t.a;

    real3 v12ref = tref.b - tref.a;
    real3 v13ref = tref.c - tref.a;

    real area_inv  = 1.0 / area(t);
    real area0_inv = 1.0 / area(tref);

    real e0sq_A = dot(v12, v12) * area_inv;
    real e1sq_A = dot(v13, v13) * area_inv;

    real e0sq_A0 = dot(v12ref, v12ref) * area0_inv;
    real e1sq_A0 = dot(v13ref, v13ref) * area0_inv;

    real v12v13    = dot(v12, v13);
    real v12v13ref = dot(v12ref, v13ref);
    
    real beta = 0.125 * (e0sq_A0*e1sq_A + e1sq_A0*e0sq_A
                         - 2. * v12v13 * v12v13ref * area_inv * area0_inv
                         - 8.);
    return beta;
}

using InvariantFunction = std::function<real(Triangle, Triangle)>;

static void testManyTriangles(long seed, int ntests, InvariantFunction invariant)
{
    constexpr real tolerance = 1e-6;

    for (int i = 0; i < ntests; ++i)
    {
        auto t    = genTriangle(seed * 42 + i);
        auto tref = genTriangle(seed + 42 * i);

        real a1, a2, a3;
        
        a1 = invariant(t, tref); shuffle(t); shuffle(tref);
        a2 = invariant(t, tref); shuffle(t); shuffle(tref);
        a3 = invariant(t, tref);
        
        ASSERT_LE(fabs(a1-a2), tolerance);
        ASSERT_LE(fabs(a1-a3), tolerance);
    }
}

TEST (TRINAGLE_STRAIN_INVARIANTS, alpha)
{
    testManyTriangles(4242, 10000, &alpha);
}

TEST (TRINAGLE_STRAIN_INVARIANTS, beta)
{
    testManyTriangles(4242, 10000, &beta_stable);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
