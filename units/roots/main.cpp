#include <core/logger.h>
#include <core/utils/root_finder.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

Logger logger;

inline float sqrtNewton(float a)
{
    auto f      = [&](float x) {return x*x - a;};
    auto fprime = [&](float x) {return 2*x;};
    
    const auto root = RootFinder::newton(f, fprime, a);
    return root.x;
}

TEST (ROOTS, Newton_sqrt)
{
    constexpr float x = 2.f;
    constexpr float tol = 1e-5f;
    
    const float val = sqrtNewton(x);
    const float ref = std::sqrt(x);
    ASSERT_LE(fabs(val-ref), tol);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
