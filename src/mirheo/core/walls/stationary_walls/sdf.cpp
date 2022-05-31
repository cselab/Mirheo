// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "sdf.h"

namespace mirheo
{

StationaryWallSDF::StationaryWallSDF(const MirState *state, std::string sdfFileName, real3 sdfH, real3 margin) :
    impl_(std::make_unique<FieldFromFile>(state, "field_"+sdfFileName, sdfFileName, sdfH, margin))
{}

StationaryWallSDF::StationaryWallSDF(StationaryWallSDF&&) = default;

const FieldDeviceHandler& StationaryWallSDF::handler() const
{
    return impl_->handler();
}

void StationaryWallSDF::setup(MPI_Comm& comm, __UNUSED DomainInfo domain)
{
    return impl_->setup(comm);
}

} // namespace mirheo
