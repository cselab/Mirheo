#pragma once

#include <functional>
#include <string>
#include <core/xml/pugixml.hpp>

class ParticleVector;
class CellList;

//==================================================================================================================
// DPD interactions
//==================================================================================================================

class Interaction
{
	enum InteractionType { Regular, Halo };

public:
	float rc;
	std::string name;

	virtual void compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) = 0;

	void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		if (pv1->local()->size() < pv2->local()->size())
			compute(InteractionType::Regular, pv1, pv2, cl1, t, stream);
		else
			compute(InteractionType::Regular, pv2, pv1, cl2, t, stream);
	}

	void halo(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		compute(InteractionType::Halo, pv1, pv2, cl1, t, stream);

		if(pv1 != pv2)
			compute(InteractionType::Halo, pv2, pv1, cl2, t, stream);
	}

	virtual ~Interaction();
};


class InteractionDPD : Interaction
{
	float a, gamma, sigma, power;

public:
	void compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream);

	InteractionDPD(pugi::xml_node node);

	~InteractionDPD() = default;
};

class InteractionLJ_objectAware : Interaction
{
	float epsilon, sigma;

public:
	void compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream);

	InteractionLJ_objectAware(pugi::xml_node node);

	~InteractionLJ_objectAware() = default;
};


class InteractionRBCMembrane : Interaction
{
	float epsilon, sigma;

public:
	void compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream);

	InteractionRBCMembrane(pugi::xml_node node);

	~InteractionRBCMembrane() = default;
};
