#pragma once

#include <mpi.h>
#include <string>
#include <mirheo/core/datatypes.h>

void writeXYZ(MPI_Comm comm, std::string fname, const real4 *positions, int np);
