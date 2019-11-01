#include "step_random_gen.h"

StepRandomGen::StepRandomGen(long seed) :
    gen(seed),
    udistr(0.001_r, 1._r)
{
    lastSample = udistr(gen);
}

StepRandomGen::~StepRandomGen() = default;

real StepRandomGen::generate(const MirState *state)
{        
    if (state->currentStep != lastIteration)
    {
        lastIteration = state->currentStep;
        lastSample    = udistr(gen);
    }
    
    return lastSample;
}
    
std::ofstream& operator<<(std::ofstream& stream, const StepRandomGen& srg)
{
    stream << srg.lastIteration << std::endl
           << srg.lastSample    << std::endl
           << srg.gen           << std::endl;
    return stream;
}

std::ifstream& operator>>(std::ifstream& stream, StepRandomGen& srg)
{
    stream >> srg.lastIteration
           >> srg.lastSample
           >> srg.gen;
    return stream;
}
