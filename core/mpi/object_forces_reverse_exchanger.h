#pragma once

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;
class ObjectHaloExchanger;

class ObjectForcesReverseExchanger : public ParticleExchanger
{
protected:
	std::vector<ObjectVector*> objects;
	ObjectHaloExchanger* entangledHaloExchanger;
	PinnedBuffer<int> sizes;

	virtual void prepareData(int id, cudaStream_t stream);
	virtual void combineAndUploadData(int id, cudaStream_t stream);

public:
	ObjectForcesReverseExchanger(MPI_Comm& comm, ObjectHaloExchanger* entangledHaloExchanger) :
		ParticleExchanger(comm), entangledHaloExchanger(entangledHaloExchanger)
	{ }

	void attach(ObjectVector* ov);

	virtual ~ObjectForcesReverseExchanger() = default;
};
