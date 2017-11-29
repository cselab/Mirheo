#pragma once
#include "interface.h"
#include <core/containers.h>
#include <core/xml/pugixml.hpp>

class SDFWall;

template<class InsideWallChecker>
class MCMCSampler : public Interaction
{
protected:
	ParticleVector* combined;
	CellList* combinedCL;

	PinnedBuffer<int> nAccepted, nRejected, nDst;
	PinnedBuffer<double> totE;
	float proposalFactor;

	float minVal, maxVal;
	const InsideWallChecker& insideWallChecker;

public:
	float a, kbT, power;

	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	MCMCSampler(std::string name,
			float rc, float a, float kbT, float power,
			float minVal, float maxVal, const InsideWallChecker& insideWallChecker);

	~MCMCSampler() = default;
};
