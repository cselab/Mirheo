#pragma once

#include <random>

class YmrState;

class StepRandomGen
{
public:
    StepRandomGen(long seed);
    ~StepRandomGen();
    
    float generate(const YmrState *state);
    
private:
    int lastIteration {-1};
    float lastSample;
    std::mt19937 gen;
    std::uniform_real_distribution<float> udistr;
};
