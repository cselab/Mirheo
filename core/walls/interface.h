#pragma once

#include <mpi.h>
#include <string>
#include <vector>

class ParticleVector;
class CellList;

class Wall
{
public:
	std::string name;

protected:
	std::vector<ParticleVector*> particleVectors;
	std::vector<CellList*> cellLists;

public:
	Wall(std::string name) : name(name) {};

	virtual void setup(MPI_Comm& comm, float3 globalDomainSize, float3 globalDomainStart, float3 localDomainSize) = 0;

	virtual void removeInner(ParticleVector* pv) = 0;
	virtual void attach(ParticleVector* pv, CellList* cl, bool check) = 0;
	virtual void bounce(float dt, cudaStream_t stream) = 0;

	virtual void check(cudaStream_t stream) = 0;

	virtual ~Wall() = default;
};
