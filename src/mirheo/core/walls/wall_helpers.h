// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <vector>
#include <string>
#include <mpi.h>
#include <mirheo/core/domain.h>

#include <cuda_runtime.h>

namespace mirheo
{

class SDFBasedWall;
class ParticleVector;

namespace wall_helpers
{

void freezeParticlesInWall(SDFBasedWall *wall, ParticleVector *pv, real minVal, real maxVal);
void freezeParticlesInWalls(std::vector<SDFBasedWall*> walls, ParticleVector *pv, real minVal, real maxVal);

void dumpWalls2XDMF(std::vector<SDFBasedWall*> walls, real3 gridH, DomainInfo domain, std::string filename, MPI_Comm cartComm);

double volumeInsideWalls(std::vector<SDFBasedWall*> walls, DomainInfo domain, MPI_Comm comm, long nSamplesPerRank);

} // namespace wall_helpers

} // namespace mirheo
