#pragma once

#include <fstream>
#include <random>

class YmrState;

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
    
    float generate(const YmrState *state);

    friend std::ofstream& operator<<(std::ofstream& stream, const StepRandomGen& gen);
    friend std::ifstream& operator>>(std::ifstream& stream,       StepRandomGen& gen);
    
private:
    int lastIteration {-1};
    float lastSample;
    std::mt19937 gen;
    std::uniform_real_distribution<float> udistr;
};
