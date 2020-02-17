#pragma once

#include <mirheo/core/field/from_file.h>
#include <memory>

namespace mirheo
{

class StationaryWallSDF
{
public:
    StationaryWallSDF(const MirState *state, std::string sdfFileName, real3 sdfH);
    StationaryWallSDF(StationaryWallSDF&&);

    void setup(MPI_Comm& comm, DomainInfo domain);

    const FieldDeviceHandler& handler() const;

private:
    std::unique_ptr<FieldFromFile> impl_;
};

} // namespace mirheo
