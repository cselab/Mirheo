#pragma once
#include "interface.h"
#include <core/xml/pugixml.hpp>

class InteractionLJ : public Interaction
{
	float epsilon, sigma;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	InteractionLJ(std::string name, float rc, float sigma, float epsilon);

	~InteractionLJ() = default;
};

class InteractionLJ_objectAware : public Interaction
{
	float epsilon, sigma;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	InteractionLJ_objectAware(std::string name, float rc, float sigma, float epsilon);

	~InteractionLJ_objectAware() = default;
};

