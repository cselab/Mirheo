#pragma once

#include <mpi.h>
#include <string>
#include <core/datatypes.h>

void writeXYZ(MPI_Comm comm, std::string fname, Particle* particles, int np);
