#pragma once

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;

class ObjectHaloExchanger : public ParticleExchanger
{
	friend class ObjectForcesReverseExchanger;

protected:
	std::vector<float> rcs;
	std::vector<ObjectVector*> objects;

	void prepareData(int id, cudaStream_t stream) override;
	void combineAndUploadData(int id, cudaStream_t stream) override;

public:
	ObjectHaloExchanger(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ObjectVector* ov, float rc);

	virtual ~ObjectHaloExchanger() = default;
};
