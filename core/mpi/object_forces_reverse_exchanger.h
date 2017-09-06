#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;

class ObjectForcesReverseExchanger : public ParticleExchanger
{
private:
	std::vector<ObjectVector*> objects;
	std::vector<int*> offsetPtrs;

	virtual void prepareData(int id, cudaStream_t stream);
	virtual void combineAndUploadData(int id);

public:
	ObjectForcesReverseExchanger(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ObjectVector* ov, int* offsetPtr);

	virtual ~ObjectForcesReverseExchanger() = default;
};
