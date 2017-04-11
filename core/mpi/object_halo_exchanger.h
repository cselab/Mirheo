#pragma once

#include <core/datatypes.h>
#include <core/logger.h>

#include <core/mpi/halo_exchanger.h>

#include <vector>

class ObjectVector;
class ParticleVector;

class ObjectHaloExchanger : public HaloExchanger
{
private:
	std::vector<float> rcs;
	std::vector<ObjectVector*> objects;

	void prepareForces(ObjectVector* ov, HaloHelper* helper);
	void uploadForces (ObjectVector* ov, HaloHelper* helper);

public:
	ObjectHaloExchanger(MPI_Comm& comm, cudaStream_t defStream) : HaloExchanger(comm, defStream) {};

	void _prepareHalos(int id);
	void attach(ObjectVector* ov, float rc);
	void exchangeForces();

	virtual ~ObjectHaloExchanger() = default;
};
