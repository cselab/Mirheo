#pragma once

#include <mpi.h>
#include <string>
#include <mirheo/core/datatypes.h>

namespace mirheo
{

void writeXYZ(MPI_Comm comm, std::string fname, const real4 *positions, int np);

} // namespace mirheo
