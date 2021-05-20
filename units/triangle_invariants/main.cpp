#include <mirheo/core/logger.h>
#include <mirheo/core/utils/helper_math.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>

using namespace mirheo;

using Real = double;
using Real3 = double3;

struct Triangle
{
    Real3 a, b, c;
};

static Triangle genTriangle(long seed)
{
    Triangle t;
    srand48(seed);

    auto gen3 = []() -> Real3 { return {(Real) drand48(), (Real) drand48(), (Real) drand48()}; };

    t.a = gen3();
    t.b = gen3();
    t.c = gen3();

    return t;
}

static void shuffle(Triangle& t)
{
    auto tmp = t.a;
    t.a = t.b;
    t.b = t.c;
    t.c = tmp;
}

static Real area(Triangle t)
{
    Real3 v12 = t.b - t.a;
    Real3 v13 = t.c - t.a;
    Real3 n = cross(v12, v13);
    return 0.5 * length(n);
}

static Real alpha(Triangle t, Triangle tref)
{
    return area(t) / area(tref) - 1;
}

inline Real safeSqrt(Real a)
{
    return a > 0.0 ? math::sqrt(a) : 0.0;
}

static Real beta_stable(Triangle t, Triangle tref)
{
    Real3 v12 = t.b - t.a;
    Real3 v13 = t.c - t.a;

    Real3 v12ref = tref.b - tref.a;
    Real3 v13ref = tref.c - tref.a;

    Real area_inv  = 1.0 / area(t);
    Real area0_inv = 1.0 / area(tref);

    Real e0sq_A = dot(v12, v12) * area_inv;
    Real e1sq_A = dot(v13, v13) * area_inv;

    Real e0sq_A0 = dot(v12ref, v12ref) * area0_inv;
    Real e1sq_A0 = dot(v13ref, v13ref) * area0_inv;

    Real v12v13    = dot(v12, v13);
    Real v12v13ref = dot(v12ref, v13ref);

    Real beta = 0.125 * (e0sq_A0*e1sq_A + e1sq_A0*e0sq_A
                         - 2. * v12v13 * v12v13ref * area_inv * area0_inv
                         - 8.);
    return beta;
}

using InvariantFunction = std::function<Real(Triangle, Triangle)>;

static void testManyTriangles(long seed, int ntests, InvariantFunction invariant)
{
    constexpr Real tolerance = 1e-6;

    for (int i = 0; i < ntests; ++i)
    {
        auto t    = genTriangle(seed * 42 + i);
        auto tref = genTriangle(seed + 42 * i);

        Real a1, a2, a3;

        a1 = invariant(t, tref); shuffle(t); shuffle(tref);
        a2 = invariant(t, tref); shuffle(t); shuffle(tref);
        a3 = invariant(t, tref);

        ASSERT_LE(math::abs(a1-a2), tolerance);
        ASSERT_LE(math::abs(a1-a3), tolerance);
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
    logger.init(MPI_COMM_NULL, "triangle_invariant.log", 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
