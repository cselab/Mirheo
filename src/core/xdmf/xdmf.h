#pragma once

#include <vector>
#include <string>
#include <memory>
#include <mpi.h>

#include <core/pvs/particle_vector.h>

#include "grids.h"

namespace XDMF
{
    void write(std::string filename, const Grid* grid, const std::vector<Channel>& channels, float time, MPI_Comm comm);
    void write(std::string filename, const Grid* grid, const std::vector<Channel>& channels, MPI_Comm comm);

    std::pair<std::unique_ptr<Grid>, std::vector<Channel>> read(std::string filename, MPI_Comm comm);
    void read(std::string filename, MPI_Comm comm, ParticleVector* pv);
}
