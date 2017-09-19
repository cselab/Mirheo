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

private:
	std::vector<ParticleVector*> particleVectors;
	std::vector<CellList*> cellLists;

public:
	Wall(std::string name) : name(name) {};

	virtual void removeInner(ParticleVector* pv) = 0;
	virtual void attach(ParticleVector* pv, CellList* cl) = 0;
	virtual void bounce(float dt, cudaStream_t stream) = 0;

	virtual void check(cudaStream_t stream) = 0;

	virtual ~Wall() = default;
};
