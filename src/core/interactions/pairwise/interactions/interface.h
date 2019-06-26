#pragma once

#include <core/mirheo_state.h>

class PairwiseKernel
{
public:
    virtual void setup(LocalParticleVector *lpv1,
                       LocalParticleVector *lpv2,
                       CellList *cl1,
                       CellList *cl2,
                       const MirState *state)
    {}
    
    virtual void writeState(std::ofstream& fout)
    {}
    
    virtual bool readState(std::ifstream& fin)
    {
        return true;
    }
};
