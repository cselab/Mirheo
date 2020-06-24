#pragma once

#include <mirheo/core/mirheo_state.h>

#include <iosfwd>
#include <random>

namespace mirheo
{

/** \brief A random number generator that generates a different number at every time step
    but returns the same number while the time step is not updated.

    Used to generate independant random numbers at  every time step.
    Several calls at the same time step will return the same random number.
    This is used to keep the interactionssymmetric accross ranks (pairwise particle halo
    interactions are computed twice, once on each rank. The random seed must therefore be
    the same and only depend on the time step, not the rank).
 */
class StepRandomGen
{
public:
    /** construct a StepRandomGen
        \param [in] seed The random seed.
    */
    explicit StepRandomGen(long seed);
    ~StepRandomGen();

    /** Generates a random number from the current state.
        \param [in] state The currenst state that contains time step info.
        \return a random number uniformly distributed on [0.001, 1].
    */
    real generate(const MirState *state);

    /// serialization helper
    friend std::ofstream& operator<<(std::ofstream& stream, const StepRandomGen& gen);
    /// deserialization helper
    friend std::ifstream& operator>>(std::ifstream& stream,       StepRandomGen& gen);

private:
    MirState::TimeType lastTime {-1};
    real lastSample;
    std::mt19937 gen;
    std::uniform_real_distribution<real> udistr;
};

} // namespace mirheo
