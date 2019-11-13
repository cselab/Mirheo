#include <mirheo/core/utils/quaternion.h>
#include <mirheo/core/logger.h>

#include <gtest/gtest.h>

#include <random>

using namespace mirheo;

constexpr real3 ex {1.0_r, 0.0_r, 0.0_r};
constexpr real3 ey {0.0_r, 1.0_r, 0.0_r};
constexpr real3 ez {0.0_r, 0.0_r, 1.0_r};

static inline void assertEquals(real3 a, real3 b, real eps = 1e-6_r)
{
    ASSERT_NEAR(a.x, b.x, eps);
    ASSERT_NEAR(a.y, b.y, eps);
    ASSERT_NEAR(a.z, b.z, eps);
}

TEST (QUATERNION, rotate_identity )
{
    auto q = Quaternion<real>::createFromComponents(1, 0, 0, 0);
    auto v = q.rotate(ex);
    assertEquals(v, ex); 
}

TEST (QUATERNION, rotate_around_axis )
{
    {
        const auto q = Quaternion<real>::createFromRotation(M_PI, ey);
        const auto v = q.rotate(ex);
        
        assertEquals(v, -ex); 
    }
    {
        const auto q = Quaternion<real>::createFromRotation(M_PI/2.0_r, ez);
        const auto v = q.rotate(ex);
        
        assertEquals(v, ey);
    }
}

static inline real3 makeRandomUnitVector(std::mt19937& gen)
{
     std::uniform_real_distribution<real> U(0.0_r, 1.0_r);
     const real theta = 2.0_r * M_PI * U(gen);
     const real phi   = std::acos(1.0_r - 2.0_r * U(gen));

     return {std::sin(phi) * std::cos(theta),
             std::sin(phi) * std::sin(theta),
             std::cos(phi)};
}

TEST (QUATERNION, construction_random )
{
    const unsigned long seed = 424242;
    const int numTries = 50;
    std::mt19937 gen(seed);

    for (int i = 0; i < numTries; ++i)
    {
        const auto u = makeRandomUnitVector(gen);
        const auto v = makeRandomUnitVector(gen);

        const auto q = Quaternion<real>::createFromVectors(u,v);
        assertEquals(v, q.rotate(u));
    }
}

TEST (QUATERNION, construction_opposite )
{
    {
        const real3 u = ex;
        const real3 v = ex;

        const auto q = Quaternion<real>::createFromVectors(u,v);
        assertEquals(v, q.rotate(u));
    }
    {
        const real3 u = ex;
        const real3 v = {-1.0_r, 0.0_r, 0.0_r};

        const auto q = Quaternion<real>::createFromVectors(u,v);
        assertEquals(v, q.rotate(u));
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
