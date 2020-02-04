#include "sdf.h"

namespace mirheo
{

StationaryWall_SDF::StationaryWall_SDF(const MirState *state, std::string sdfFileName, real3 sdfH) :
    impl_(std::make_unique<FieldFromFile>(state, "field_"+sdfFileName, sdfFileName, sdfH))
{}

StationaryWall_SDF::StationaryWall_SDF(StationaryWall_SDF&&) = default;

const FieldDeviceHandler& StationaryWall_SDF::handler() const
{
    return impl_->handler();
}

void StationaryWall_SDF::setup(MPI_Comm& comm, __UNUSED DomainInfo domain)
{
    return impl_->setup(comm);
}

} // namespace mirheo
