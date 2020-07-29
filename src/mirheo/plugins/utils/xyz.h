// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mpi.h>
#include <string>
#include <mirheo/core/datatypes.h>

namespace mirheo
{

/** Dump positions to a file in xyz format using MPI IO.
    \param [in] comm The MPI communicator.
    \param [in] fname The name of the target file.
    \param [in] positions Array of positions xyz_.
    \param [in] np Local number of particles.
 */
void writeXYZ(MPI_Comm comm, std::string fname, const real4 *positions, int np);

} // namespace mirheo
