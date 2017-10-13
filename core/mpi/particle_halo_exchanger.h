#pragma once

#include "particle_exchanger.h"

#include <vector>

class ParticleVector;
class CellList;

class ParticleHaloExchanger : public ParticleExchanger
{
private:
	std::vector<CellList*> cellLists;
	std::vector<ParticleVector*> particles;

	void prepareData(int id, cudaStream_t stream);
	void combineAndUploadData(int id, cudaStream_t stream);
	bool needExchange(int id) override;

public:
	ParticleHaloExchanger(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ParticleVector* pv, CellList* cl);

	~ParticleHaloExchanger() = default;
};
