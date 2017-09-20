#pragma once

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;

class ObjectRedistributor : public ParticleExchanger
{
private:
	std::vector<ObjectVector*> objects;

	virtual void prepareData(int id, cudaStream_t stream);
	virtual void combineAndUploadData(int id, cudaStream_t stream);

public:
	ObjectRedistributor(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ObjectVector* ov, float rc);
	void redistribute();

	virtual ~ObjectRedistributor() = default;
};
