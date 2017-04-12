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

	void prepareForces(ObjectVector* ov, ExchangeHelper* helper);
	void uploadForces (ObjectVector* ov, ExchangeHelper* helper);

	void prepareUploadTarget(int id);
	void prepareData(int id);

public:
	ObjectHaloExchanger(MPI_Comm& comm, cudaStream_t defStream) : ParticleExchanger(comm, defStream) {};

	void attach(ObjectVector* ov, float rc);
	void exchangeForces();

	virtual ~ObjectHaloExchanger() = default;
};
