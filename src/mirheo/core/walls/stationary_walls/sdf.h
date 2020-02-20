#pragma once

#include <mirheo/core/field/from_file.h>
#include <memory>

namespace mirheo
{

/// Represent an arbitrary SDF field on a grid.
class StationaryWallSDF
{
public:
    /** \brief Construct a StationaryWallSDF from a file.
        \param [in] state Simulation state
        \param [in] sdfFileName The input file name
        \param [in] sdfH The grid spacing
     */
    StationaryWallSDF(const MirState *state, std::string sdfFileName, real3 sdfH);
    /// Move ctor.
    StationaryWallSDF(StationaryWallSDF&&);

    /** \brief Synchronize internal state with simulation
        \param [in] comm MPI carthesia communicator
        \param [in] domain Domain info
    */
    void setup(MPI_Comm& comm, DomainInfo domain);

    /// Get a handler of the shape representation usable on the device
    const FieldDeviceHandler& handler() const;

private:
    std::unique_ptr<FieldFromFile> impl_;
};

} // namespace mirheo
