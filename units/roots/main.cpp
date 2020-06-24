#include <mirheo/core/logger.h>
#include <mirheo/core/utils/root_finder.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

#include <random>

using namespace mirheo;

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
        const float ref = math::sqrt(x);
        ASSERT_LE(math::abs(val-ref), tol) << "estimate of sqrt(" << x << ") : got " << val << ", ref is " << ref;
    }
}

TEST (ROOTS, Newton_sqrt)
{
    auto sqrtSolverNewton = [](float a)
    {
        auto f      = [&](float x) {return x*x - a;};
        auto fprime = [&](float x) {return 2*x;};

        const auto root = root_finder::newton(f, fprime, a);
        return root.x;
    };

    testSolver(sqrtSolverNewton);
}

TEST (ROOTS, LinearSearch_sqrt)
{
    auto sqrtSolverLinearSearch = [](float a)
    {
        auto f = [&](float x) {return x*x - a;};

        const root_finder::Bounds limits{0.f, a};

        const auto root = root_finder::linearSearchVerbose(f, limits);
        return root.x;
    };

    testSolver(sqrtSolverLinearSearch);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
