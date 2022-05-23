#include <mirheo/core/domain.h>
#include <mirheo/core/interactions/utils/step_random_gen.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/mirheo_state.h>
#include <mirheo/core/utils/cuda_rng.h>

#include <gtest/gtest.h>

using namespace mirheo;

template<typename Gen>
static std::vector<real> generateSamples(Gen gen, real dt, long n)
{
    DomainInfo domain;
    std::vector<real> samples (n);
    MirState state(domain, dt);
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
    MirState state(DomainInfo{}, dt);
    state.currentTime = 1243234;

    const auto x0 = gen.generate(&state);
    const auto x1 = gen.generate(&state);

    ASSERT_EQ(x0, x1);
}



TEST (Saru, uniform01_is_within_bounds)
{
    const int n = 10000;
    const real seed = 0.138768175_r;

    for (int i = 0; i < n; ++i)
    {
        const int j = -35 * i + i*i + 3;

        const real x = Saru::uniform01(seed, i, j);

        ASSERT_LE(0.0_r, x);
        ASSERT_LE(x, 1.0_r);
    }
}

TEST (Saru, mean0var1_has_zero_mean)
{
    const int n = 100000;
    const real seed = 0.138768175_r;

    real mean = 0;

    for (int i = 0; i < n; ++i)
    {
        const int j = -35 * i + i*i + 3;
        const real x = Saru::mean0var1(seed, i, j);

        mean += x;
    }

    mean /= n;
    ASSERT_NEAR(mean, 0.0_r, 2.0 / std::sqrt(n));
}

TEST (Saru, mean0var1_has_unit_variance)
{
    const int n = 100000;
    const real seed = 0.138768175_r;

    const real mean = 0;
    real var = 0;

    for (int i = 0; i < n; ++i)
    {
        const int j = -35 * i + i*i + 3;
        const real x = Saru::mean0var1(seed, i, j);
        const real dx = x - mean;
        var += dx*dx;
    }

    var /= (n-1);
    ASSERT_NEAR(var, 1.0_r, 2.0 / std::sqrt(n));
}





int main(int argc, char **argv)
{
    logger.init(MPI_COMM_NULL, "rng.log", 0);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
