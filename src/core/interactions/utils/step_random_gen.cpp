#include "step_random_gen.h"

#include <core/ymero_state.h>

StepRandomGen::StepRandomGen(long seed) :
    gen(seed),
    udistr(0.001f, 1.f)
{
    lastSample = udistr(gen);
}

StepRandomGen::~StepRandomGen() = default;

float StepRandomGen::generate(const YmrState *state)
{        
    if (state->currentStep != lastIteration)
    {
        lastIteration = state->currentStep;
        lastSample = udistr(gen);
    }
    
    return lastSample;
}
    
