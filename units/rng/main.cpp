#include <mirheo/core/logger.h>
#include <mirheo/core/interactions/utils/step_random_gen.h>
#include <mirheo/core/domain.h>
#include <mirheo/core/mirheo_state.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <random>

using namespace mirheo;

template<typename Gen>
static std::vector<real> generateSamples(Gen gen, real dt, long n)
{
    DomainInfo domain;
    std::vector<real> samples (n);
    MirState state(domain, dt, UnitConversion{});
    state.currentTime = 0;

    for (state.currentStep = 0; state.currentStep < n; ++state.currentStep)
    {
        samples[state.currentStep] = gen.generate(&state);
        state.currentTime += state.getDt();
    }

    return samples;
}

using Real = long double;

template<typename Gen>
static Real computeAutoCorrelation(Gen gen, real dt, long n)
{
    const auto samples = generateSamples(gen, dt, n);

    Real sum = 0;
    for (const auto& x : samples)
        sum += (Real) x;

    const Real mean = sum / n;
    const Real mean_sq = mean*mean;

    Real covariance = 0;
    for (int i = 1; i < n; ++i)
        covariance += samples[i] * samples[i-1] - mean_sq;

    return covariance / n;
}

TEST (StepRandomGen, auto_correlation_is_zero)
{
    StepRandomGen gen(424242);
    const real dt = 1e-3;
    const Real corr = computeAutoCorrelation(gen, dt, 10000);
    ASSERT_LE(std::abs(corr), 2e-3);
}

TEST (StepRandomGen, gives_same_value_at_same_time)
{
    StepRandomGen gen(424242);
    const real dt = 1e-3;
    MirState state(DomainInfo{}, dt, UnitConversion{});
    state.currentTime = 1243234;

    const auto x0 = gen.generate(&state);
    const auto x1 = gen.generate(&state);

    ASSERT_EQ(x0, x1);
}

int main(int argc, char **argv)
{
    logger.init(MPI_COMM_NULL, "rng.log", 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
