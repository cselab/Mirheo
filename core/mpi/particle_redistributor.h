#pragma once

#include "particle_exchanger.h"

#include <vector>

class ParticleVector;
class CellList;

class ParticleRedistributor : public ParticleExchanger
{
private:
	std::vector<ParticleVector*> particles;
	std::vector<CellList*> cellLists;

	void combineAndUploadData(int id, cudaStream_t stream) override;
	void prepareData(int id, cudaStream_t stream) override;
	bool needExchange(int id) override;

public:
	void _prepareData(int id);

	ParticleRedistributor(MPI_Comm& comm) : ParticleExchanger(comm) {};
	void attach(ParticleVector* pv, CellList* cl);

	~ParticleRedistributor() = default;
};
