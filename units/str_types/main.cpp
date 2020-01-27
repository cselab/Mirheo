#include <mirheo/core/logger.h>
#include <mirheo/core/types/str.h>

#include <gtest/gtest.h>

using namespace mirheo;

TEST (STR_TYPES, int)
{
    const int i = 42; 
    ASSERT_EQ(printToStr(i), "42");
}

TEST (STR_TYPES, int64)
{
    const int64_t i = 42; 
    ASSERT_EQ(printToStr(i), "42");
}

TEST (STR_TYPES, float)
{
    const float f = 42.3f; 
    ASSERT_EQ(printToStr(f), "42.3");
}

TEST (STR_TYPES, float2)
{
    const float2 f {1.1f, 2.2f}; 
    ASSERT_EQ(printToStr(f), "1.1 2.2");
}

TEST (STR_TYPES, float3)
{
    const float3 f {1.1f, 2.2f, 3.3f}; 
    ASSERT_EQ(printToStr(f), "1.1 2.2 3.3");
}

TEST (STR_TYPES, float4)
{
    const float4 f {1.1f, 2.2f, 3.3f, 4.4f}; 
    ASSERT_EQ(printToStr(f), "1.1 2.2 3.3 4.4");
}

TEST (STR_TYPES, COMandExtent)
{
    const float3 com {1.1f, 2.2f, 3.3f};
    const float3 lo {-4.f, -5.f, -6.f};
    const float3 hi {4.f, 5.f, 6.f}; 
    const COMandExtent cae {com, lo, hi};
    
    ASSERT_EQ(printToStr(cae), "[com: 1.1 2.2 3.3, lo: -4 -5 -6, hi: 4 5 6]");
}

TEST (STR_TYPES, RigidMotion)
{
    const RigidReal3 r {1.1, 2.2, 3.3};
    const auto q = Quaternion<RigidReal>::createFromComponents(1., 0., 0., 0.);
    const RigidReal3 vel {-4., -5., -6.};
    const RigidReal3 omega {4., 5., 6.};
    const RigidReal3 f {8., 8., 8.};
    const RigidReal3 t {9., 9., 9.};

    const RigidMotion m {r, q, vel, omega, f, t};
    
    ASSERT_EQ(printToStr(m),
              "[r: 1.1 2.2 3.3, q: 1 0 0 0, v: -4 -5 -6, w: 4 5 6, F: 8 8 8, T: 9 9 9]");
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
