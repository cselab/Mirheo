#pragma once

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;

class ObjectHaloExchanger : public ParticleExchanger
{
protected:
	std::vector<float> rcs;
	std::vector<ObjectVector*> objects;

	std::vector<PinnedBuffer<int>*> origins;

	void prepareData(int id, cudaStream_t stream) override;
	void combineAndUploadData(int id, cudaStream_t stream) override;
	bool needExchange(int id) override;

public:
	ObjectHaloExchanger(MPI_Comm& comm) : ParticleExchanger(comm) {};

	void attach(ObjectVector* ov, float rc);

	PinnedBuffer<int>& getRecvOffsets(int id);
	PinnedBuffer<int>& getOrigins    (int id);

	virtual ~ObjectHaloExchanger() = default;
};
