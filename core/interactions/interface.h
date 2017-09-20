#pragma once

#include <functional>
#include <string>
#include <core/pvs/particle_vector.h>

class CellList;

//==================================================================================================================
// DPD interactions
//==================================================================================================================

class Interaction
{
public:
	enum class InteractionType { Regular, Halo };

public:
	float rc;
	std::string name;

	Interaction(std::string name, float rc) : name(name), rc(rc) {}

	/**
	 * This function is not supposed to be called directly.
	 * Cannot make it private because of CUDA limitations
	 */
	virtual void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream) = 0;

	void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		if (pv1->local()->size() < pv2->local()->size())
			_compute(InteractionType::Regular, pv1, pv2, cl1, t, stream);
		else
			_compute(InteractionType::Regular, pv2, pv1, cl2, t, stream);
	}

	void halo(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		_compute(InteractionType::Halo, pv1, pv2, cl1, t, stream);

		if(pv1 != pv2)
			_compute(InteractionType::Halo, pv2, pv1, cl2, t, stream);
	}

	virtual ~Interaction() = default;
};
