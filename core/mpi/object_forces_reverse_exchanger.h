#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <core/mpi/particle_exchanger.h>

#include <vector>

class ObjectVector;

class ObjectForcesReverseExchanger : public ParticleExchanger
{
private:
	std::vector<ObjectVector*> objects;
	std::vector<int*> offsetPtrs;

	virtual void prepareData(int id);
	virtual void combineAndUploadData(int id);

public:
	ObjectForcesReverseExchanger(MPI_Comm& comm, cudaStream_t defStream) : ParticleExchanger(comm, defStream) {};

	void attach(ObjectVector* ov, int* offsetPtr);

	virtual ~ObjectForcesReverseExchanger() = default;
};
