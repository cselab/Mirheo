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
    MirState state(domain, dt);
    state.currentTime = 0;    

    for (state.currentStep = 0; state.currentStep < n; ++state.currentStep)
    {
        samples[state.currentStep] = gen.generate(&state);
        state.currentTime += state.dt;
    }

    return samples;
}

using Real = long double;

template<typename Gen>
static Real computeAutoCorrelation(Gen gen, real dt, long n)
{
    auto samples = generateSamples(gen, dt, n);
    
    Real sum = 0;
    for (const auto& x : samples) sum += (Real) x;

    Real mean = sum / n;
    Real covariance = 0;
    Real mean_sq = mean*mean;

    for (int i = 1; i < n; ++i)
        covariance += samples[i] * samples[i-1] - mean_sq;

    return covariance / n;
}

class GenFromTime
{
public:
    real generate(const MirState *state)
    {
        const real t = state->currentTime;
        const real *pt = &t;
        const int v = *reinterpret_cast<const int*>(pt);
        std::mt19937 gen(v);
        std::uniform_real_distribution<real> udistr(0.001, 1);
        return udistr(gen);
    }
};

TEST (RNG, autoCorrelationGenFromTime)
{
    GenFromTime gen;
    real dt = 1e-3;
    
    auto corr = computeAutoCorrelation(gen, dt, 10000);

    printf("from time: %g\n", (double) corr);
                                       
    ASSERT_LE(std::abs(corr), 1e-3);
}

TEST (RNG, autoCorrelationGenFromMT)
{
    StepRandomGen gen(424242);
    real dt = 1e-3;
    
    auto corr = computeAutoCorrelation(gen, dt, 10000);

    printf("from MT: %g\n", (double) corr);
                                       
    ASSERT_LE(std::abs(corr), 2e-3);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
