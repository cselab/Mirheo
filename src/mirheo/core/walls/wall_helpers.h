#pragma once

#include <vector>
#include <string>
#include <mpi.h>
#include <mirheo/core/domain.h>

#include <cuda_runtime.h>

class SDF_basedWall;
class ParticleVector;

namespace WallHelpers
{

void freezeParticlesInWall(SDF_basedWall *wall, ParticleVector *pv, real minVal, real maxVal);
void freezeParticlesInWalls(std::vector<SDF_basedWall*> walls, ParticleVector *pv, real minVal, real maxVal);

void dumpWalls2XDMF(std::vector<SDF_basedWall*> walls, real3 gridH, DomainInfo domain, std::string filename, MPI_Comm cartComm);

double volumeInsideWalls(std::vector<SDF_basedWall*> walls, DomainInfo domain, MPI_Comm comm, long nSamplesPerRank);

} // namespace WallHelpers
