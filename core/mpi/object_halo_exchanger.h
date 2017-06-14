#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <core/mpi/particle_exchanger.h>

#include <vector>

class ObjectVector;

class ObjectHaloExchanger : public ParticleExchanger
{
private:
	std::vector<float> rcs;
	std::vector<ObjectVector*> objects;

	virtual void prepareData(int id);
	virtual void combineAndUploadData(int id);

public:
	ObjectHaloExchanger(MPI_Comm& comm, cudaStream_t defStream) : ParticleExchanger(comm, defStream) {};

	void attach(ObjectVector* ov, float rc);

	virtual ~ObjectHaloExchanger() = default;
};
