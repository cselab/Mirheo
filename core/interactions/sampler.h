#pragma once
#include "interface.h"
#include <core/containers.h>
#include <core/xml/pugixml.hpp>

class Wall;

class MCMCSampler : public Interaction
{
protected:
	ParticleVector* combined;
	CellList* combinedCL;
	Wall* wall;

	PinnedBuffer<int> nAccepted, nRejected, nDst;
	PinnedBuffer<double> totE;
	float proposalFactor;
	float minSdf, maxSdf;

public:
	float a, kbT, power;

	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) override;

	MCMCSampler(std::string name, float rc, float a, float kbT, float power, Wall* wall, float minSdf, float maxSdf);

	~MCMCSampler() = default;
};
