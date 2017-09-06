#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;

class ObjectRedistributor : public ParticleExchanger
{
private:
	std::vector<ObjectVector*> objects;

	virtual void prepareData(int id, cudaStream_t stream);
	virtual void combineAndUploadData(int id);

public:
	ObjectRedistributor(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ObjectVector* ov, float rc);
	void redistribute();

	virtual ~ObjectRedistributor() = default;
};
