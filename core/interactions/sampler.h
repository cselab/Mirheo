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


public:
	static constexpr float minSdf = -2.0f;
	static constexpr float maxSdf = 3.0f;


	float a, kbT, power;

	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream);

	MCMCSampler(pugi::xml_node node, Wall* wall);

	~MCMCSampler() = default;
};
