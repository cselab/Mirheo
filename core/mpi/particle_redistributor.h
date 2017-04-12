#pragma once

#include <core/datatypes.h>

#include <vector>

class ParticleVector;
class CellList;

class ParticleRedistributor : public ParticleExchanger
{
private:
	std::vector<ParticleVector*> particles;
	std::vector<CellList*> cellLists;

	void prepareUploadTarget(int id);

public:
	void _prepareData(int id);

	ParticleRedistributor(MPI_Comm& comm);
	void attach(ParticleVector* pv, CellList* cl);
	void redistribute();

	~ParticleRedistributor() = default;
};
