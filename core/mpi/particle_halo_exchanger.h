#pragma once

#include <core/datatypes.h>
#include <core/logger.h>
#include <core/mpi/halo_exchanger.h>

#include <vector>

class ParticleVector;
class CellList;

class ParticleHaloExchanger : public HaloExchanger
{
private:
	std::vector<CellList*> cellLists;
	std::vector<ParticleVector*> particles;

public:
	ParticleHaloExchanger(MPI_Comm& comm, cudaStream_t defStream) : HaloExchanger(comm, defStream) {};

	void _prepareHalos(int id);
	void attach(ParticleVector* pv, CellList* cl);

	~ParticleHaloExchanger() = default;
};
