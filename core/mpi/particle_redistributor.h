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

	void combineAndUploadData(int id);
	void prepareData(int id, cudaStream_t defStream);

public:
	void _prepareData(int id);

	ParticleRedistributor(MPI_Comm& comm) : ParticleExchanger(comm) {};
	void attach(ParticleVector* pv, CellList* cl);
	void redistribute(cudaStream_t defStream);

	~ParticleRedistributor() = default;
};
