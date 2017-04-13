#pragma once

#include <core/datatypes.h>
#include <core/mpi/particle_exchanger.h>

#include <vector>

class ParticleVector;
class CellList;

class ParticleRedistributor : public ParticleExchanger
{
private:
	std::vector<ParticleVector*> particles;
	std::vector<CellList*> cellLists;

	void prepareUploadTarget(int id);
	void prepareData(int id);

public:
	void _prepareData(int id);

	ParticleRedistributor(MPI_Comm& comm, cudaStream_t stream) : ParticleExchanger(comm, stream) {};
	void attach(ParticleVector* pv, CellList* cl);
	void redistribute();

	~ParticleRedistributor() = default;
};
