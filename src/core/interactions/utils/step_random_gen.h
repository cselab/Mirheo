#pragma once

#include <core/ymero_state.h>

#include <limits>
#include <random>

struct StepRandomGen
{
    StepRandomGen(long seed) :
        gen(seed),
        udistr(0.001f, 1.f)
    {
        lastSample = udistr(gen);
    }
    
    float generate(const YmrState *state)
    {        
        if (state->currentStep != lastIteration)
        {
            lastIteration = state->currentStep;
            lastSample = udistr(gen);
        }

        return lastSample;
    }
    
private:
    int lastIteration {-1};
    float lastSample;
    std::mt19937 gen;
    std::uniform_real_distribution<float> udistr;
};
