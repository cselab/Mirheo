#pragma once

#include <mirheo/core/mirheo_state.h>

class PairwiseKernel
{
public:
    virtual void setup(__UNUSED LocalParticleVector *lpv1,
                       __UNUSED LocalParticleVector *lpv2,
                       __UNUSED CellList *cl1,
                       __UNUSED CellList *cl2,
                       __UNUSED const MirState *state)
    {}
    
    virtual void writeState(__UNUSED std::ofstream& fout)
    {}
    
    virtual bool readState(__UNUSED std::ifstream& fin)
    {
        return true;
    }
};
