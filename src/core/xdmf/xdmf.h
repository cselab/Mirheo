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
           const std::vector<Channel>& channels, float time, MPI_Comm comm);
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels,             MPI_Comm comm);


struct VertexChannelsData
{
    std::shared_ptr<std::vector<float3>> positions;
    std::vector<Channel> descriptions;
    std::vector<std::vector<char>> data;    
};

// chunkSize: smallest piece that processors can split
VertexChannelsData readVertexData(const std::string& filename, MPI_Comm comm, int chunkSize);

void readParticleData    (std::string filename, MPI_Comm comm, ParticleVector *pv, int chunkSize = 1);
void readObjectData      (std::string filename, MPI_Comm comm, ObjectVector *ov);
void readRigidObjectData (std::string filename, MPI_Comm comm, RigidObjectVector *rov);
}
