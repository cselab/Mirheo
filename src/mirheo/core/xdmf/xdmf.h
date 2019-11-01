#pragma once

#include "grids.h"

#include <core/pvs/rigid_object_vector.h>

#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace XDMF
{
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, real time, MPI_Comm comm);
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels,             MPI_Comm comm);


struct VertexChannelsData
{
    std::vector<real3> positions;
    std::vector<Channel> descriptions;
    std::vector<std::vector<char>> data;    
};

// chunkSize: smallest piece that processors can split
VertexChannelsData readVertexData(const std::string& filename, MPI_Comm comm, int chunkSize);

} // namespace XDMF
