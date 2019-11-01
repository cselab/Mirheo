#pragma once

#include <mirheo/core/mirheo_state.h>

#include <fstream>
#include <random>

/** \brief A simple random generator wrapper for 
 *         per time step random number generation
 * 
 * Used to generate independant random numbers at 
 * every time step. Several calls at the same time 
 * step will return the same random number
 */
class StepRandomGen
{
public:
    explicit StepRandomGen(long seed);
    ~StepRandomGen();
    
    real generate(const MirState *state);

    friend std::ofstream& operator<<(std::ofstream& stream, const StepRandomGen& gen);
    friend std::ifstream& operator>>(std::ifstream& stream,       StepRandomGen& gen);
    
private:
    MirState::StepType lastIteration {-1};
    real lastSample;
    std::mt19937 gen;
    std::uniform_real_distribution<real> udistr;
};
