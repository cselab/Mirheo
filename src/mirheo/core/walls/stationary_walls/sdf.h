// Copyright 2020 ETH Zurich. All Rights Reserved.
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
        \param [in] margin Additional margin to store in each rank; useful to bounce-back local particles.
     */
    StationaryWallSDF(const MirState *state, std::string sdfFileName, real3 sdfH, real3 margin);
    /// Move ctor.
    StationaryWallSDF(StationaryWallSDF&&);

    /** \brief Synchronize internal state with simulation
        \param [in] comm MPI carthesia communicator
        \param [in] domain Domain info
    */
    void setup(MPI_Comm& comm, DomainInfo domain);

    /// Get a handler of the shape representation usable on the device
    const ScalarFieldDeviceHandler& handler() const;

private:
    std::unique_ptr<ScalarFieldFromFile> impl_;
};

} // namespace mirheo
