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

	virtual void prepareData(int id, cudaStream_t defStream);
	virtual void combineAndUploadData(int id);

public:
	ObjectHaloExchanger(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ObjectVector* ov, float rc);

	virtual ~ObjectHaloExchanger() = default;
};
