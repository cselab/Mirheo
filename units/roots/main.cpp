#include <core/logger.h>
#include <core/utils/root_finder.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

#include <random>

Logger logger;

// inline float sqrtLinSearch(float a, float left, float right)
// {
//     auto f      = [&](float x) {return x*x - a;};
//     auto fprime = [&](float x) {return 2*x;};
    
//     const auto root = RootFinder::newton(f, fprime, a);
//     return root.x;
// }



template <class Solver>
inline void testSolver(Solver sqrtSolver)
{
    constexpr int numTries {10};
    constexpr float tol = 1e-5f;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> distr(1.0f, 10.0f);

    for (int i = 0; i < numTries; ++i)
    {
        const float x = distr(gen);

        const float val = sqrtSolver(x);
        const float ref = std::sqrt(x);
        ASSERT_LE(fabs(val-ref), tol) << "estimate of sqrt(" << x << ") : got " << val << ", ref is " << ref;
    }
}

TEST (ROOTS, Newton_sqrt)
{
    auto sqrtSolverNewton = [](float a)
    {
        auto f      = [&](float x) {return x*x - a;};
        auto fprime = [&](float x) {return 2*x;};
        
        const auto root = RootFinder::newton(f, fprime, a);
        return root.x;
    };
                               
    testSolver(sqrtSolverNewton);
}

TEST (ROOTS, LinearSearch_sqrt)
{
    auto sqrtSolverLinearSearch = [](float a)
    {
        auto f = [&](float x) {return x*x - a;};

        const RootFinder::Bounds limits{0.f, a};
        
        const auto root = RootFinder::linearSearchVerbose(f, limits);
        return root.x;
    };
                               
    testSolver(sqrtSolverLinearSearch);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
