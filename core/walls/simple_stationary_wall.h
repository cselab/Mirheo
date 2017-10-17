#pragma once

#include "interface.h"

#include <core/containers.h>

class ParticleVector;
class CellList;


template<class InsideWallChecker>
class SimpleStationaryWall : public Wall
{
public:
	SimpleStationaryWall(std::string name, InsideWallChecker&& insideWallChecker) :
		Wall(name), insideWallChecker(std::move(insideWallChecker))
	{	}

	void setup(MPI_Comm& comm, DomainInfo domain) override;

	void removeInner(ParticleVector* pv) override;
	void attach(ParticleVector* pv, CellList* cl) override;
	void bounce(float dt, cudaStream_t stream) override;
	void check(cudaStream_t stream) override;

	InsideWallChecker& getChecker() { return insideWallChecker; }

private:
	MPI_Comm wallComm;

	InsideWallChecker insideWallChecker;

	std::vector<ParticleVector*> particleVectors;
	std::vector<CellList*> cellLists;

	std::vector<int> nBounceCalls;

	std::vector<DeviceBuffer<int>*> boundaryCells;
	PinnedBuffer<int> nInside{1};
};
