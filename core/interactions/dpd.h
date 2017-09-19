#pragma once
#include "interface.h"
#include <core/xml/pugixml.hpp>

class InteractionDPD : public Interaction
{
	float a, gamma, sigma, power;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) override;

	InteractionDPD(std::string name, float rc, float a, float gamma, float kbT, float dt, float power);

	~InteractionDPD() = default;
};
