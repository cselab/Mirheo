#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <core/mpi/particle_exchanger.h>

#include <vector>

class ObjectVector;

class ObjectRedistributor : public ParticleExchanger
{
private:
	std::vector<ObjectVector*> objects;

	virtual void prepareData(int id);
	virtual void combineAndUploadData(int id);

public:
	ObjectRedistributor(MPI_Comm& comm, cudaStream_t defStream) : ParticleExchanger(comm, defStream) {};

	void attach(ObjectVector* ov, float rc);
	void redistribute();

	virtual ~ObjectRedistributor() = default;
};
