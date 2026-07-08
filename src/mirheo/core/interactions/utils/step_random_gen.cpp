// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "step_random_gen.h"
#include <fstream>

namespace mirheo
{

StepRandomGen::StepRandomGen(long seed) :
    gen(seed),
    udistr(0.001_r, 1._r)
{
    lastSample = udistr(gen);
}

real StepRandomGen::generate(const MirState *state)
{
    if (state->currentTime != lastTime)
    {
        lastTime   = state->currentTime;
        lastSample = udistr(gen);
    }

    return lastSample;
}

std::ofstream& operator<<(std::ofstream& stream, const StepRandomGen& srg)
{
    stream << srg.lastTime    << std::endl
           << srg.lastSample  << std::endl
           << srg.gen         << std::endl;
    return stream;
}

std::ifstream& operator>>(std::ifstream& stream, StepRandomGen& srg)
{
    stream >> srg.lastTime
           >> srg.lastSample
           >> srg.gen;
    return stream;
}

} // namespace mirheo
