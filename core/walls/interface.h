#pragma once

#include <mpi.h>
#include <string>
#include <vector>

#include <core/domain.h>

class ParticleVector;
class CellList;

class Wall
{
public:
	std::string name;

	Wall(std::string name) : name(name) {};

	virtual void setup(MPI_Comm& comm, DomainInfo domain, ParticleVector* jointPV) = 0;

	virtual void removeInner(ParticleVector* pv) = 0;
	virtual void attach(ParticleVector* pv, CellList* cl) = 0;
	virtual void bounce(float dt, cudaStream_t stream) = 0;

	/**
	 * Ask ParticleVectors which the class will be working with to have specific properties
	 * Default: ask nothing
	 * Called from Simulation right after setup
	 */
	virtual void setPrerequisites(ParticleVector* pv) {}

	virtual void check(cudaStream_t stream) = 0;

	virtual ~Wall() = default;
};
