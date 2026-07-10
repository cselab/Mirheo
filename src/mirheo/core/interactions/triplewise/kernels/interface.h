// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/mirheo_state.h>

namespace mirheo
{

/// Interface of host methods required for a triplewise kernel
class TriplewiseKernelBase
{
public:
    /// setup the internal state of the functor
    void setup(__UNUSED CellList *cl1,
               __UNUSED CellList *cl2,
               __UNUSED CellList *cl3,
               __UNUSED const MirState *state)
    {}

    /// write internal state to a stream
    void writeState(__UNUSED std::ofstream& fout)
    {}

    /// restore internal state from a stream
    bool readState(__UNUSED std::ifstream& fin)
    {
        return true;
    }
};

} // namespace mirheo
