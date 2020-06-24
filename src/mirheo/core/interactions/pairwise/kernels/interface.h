#pragma once

#include <mirheo/core/mirheo_state.h>

namespace mirheo
{

/// Interface of host methods required for a pairwise kernel
class PairwiseKernel
{
public:
    /// setup the internal state of the functor
    virtual void setup(__UNUSED LocalParticleVector *lpv1,
                       __UNUSED LocalParticleVector *lpv2,
                       __UNUSED CellList *cl1,
                       __UNUSED CellList *cl2,
                       __UNUSED const MirState *state)
    {}

    /// write internal state to a stream
    virtual void writeState(__UNUSED std::ofstream& fout)
    {}

    /// restore internal state from a stream
    virtual bool readState(__UNUSED std::ifstream& fin)
    {
        return true;
    }
};

} // namespace mirheo
