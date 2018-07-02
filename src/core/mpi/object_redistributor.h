#pragma once

#include "particle_exchanger.h"

#include <vector>

class ObjectVector;

class ObjectRedistributor : public ParticleExchanger
{
private:
	std::vector<ObjectVector*> objects;

	void prepareSizes(int id, cudaStream_t stream) override;
	void prepareData (int id, cudaStream_t stream) override;
	void combineAndUploadData(int id, cudaStream_t stream) override;
	bool needExchange(int id) override;

public:
	ObjectRedistributor(MPI_Comm& comm, bool gpuAwareMPI) : ParticleExchanger(comm, gpuAwareMPI) {};

	void attach(ObjectVector* ov, float rc);
	void redistribute();

	virtual ~ObjectRedistributor() = default;
};
