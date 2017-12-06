#pragma once

#include <functional>
#include <string>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>

class CellList;

/**
 * Interface for classes computing particle interactions.
 *
 * At the moment cut-off radius is the part of the interface,
 * so every interaction will require cell-list creation.
 * The cut-off raduis has to be removed later from the interface,
 * such that only certain interactions require cell-lists.
 */
class Interaction
{
public:
	enum class InteractionType { Regular, Halo };

public:
	/// Cut-off raduis
	float rc;
	std::string name;

	Interaction(std::string name, float rc) : name(name), rc(rc) {}

	/**
	 * Ask ParticleVectors which the class will be working with to have specific properties
	 * Default: ask nothing
	 * Called from Simulation right after setup
	 */
	virtual void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) {}

	/**
	 * This is the function that derived classes need to implement
	 * to compute the interactions.
	 *
	 * It is not supposed to be called directly, but cannot be made
	 * private because of CUDA limitations.
	 *
	 * If computing Halo interactions, \p pv1 will be treated as halo, and
	 * \p pv2 -- as local.
	 *
	 * @param type whether local or halo interactions need to be computed
	 * @param pv1 first interacting ParticleVector
	 * @param pv2 second interacting ParticleVector. If it is the same as
	 *            the \p pv1, self interactions will be computed
	 * @param cl1 cell-list built for the appropriate cut-off raduis #rc for \p pv1
	 * @param cl2 cell-list built for the appropriate cut-off raduis #rc for \p pv2
	 * @param t current simulation time
	 */
	virtual void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) = 0;

	/**
	 * Interface to _compute() with local interaction type.
	 * For now order of \e pv1 and \e pv2 is important for computational reasons,
	 * this may be changed later on so that the best order is chosen automatically.
	 */
	void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		//if (pv1->local()->size() < pv2->local()->size())
			_compute(InteractionType::Regular, pv1, pv2, cl1, cl2, t, stream);
		//else
		//	_compute(InteractionType::Regular, pv2, pv1, cl2, cl1, t, stream);
	}

	/**
	 * Interface to _compute() with halo interaction type.
	 *
	 * The following cases exist:
	 * - If one of \p pv1 or \p pv2 is ObjectVector, then only call to the _compute()
	 *   needed: for halo ObjectVector another ParticleVector (or ObjectVector).
	 *   This is because ObjectVector will collect the forces from remote processors,
	 *   so we don't need to compute them twice.
	 *
	 * - Both are ParticleVector. Then if they are different, two _compute() calls
	 *   are made such that halo1 \<-\> local2 and halo2 \<-\> local1. If \p pv1 and
	 *   \p pv2 are the same, only one call is needed
	 */
	void halo(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
	{
		auto isov1 = dynamic_cast<ObjectVector*>(pv1) == nullptr;
		auto isov2 = dynamic_cast<ObjectVector*>(pv2) == nullptr;

		// Two object vectors. Compute just one interaction, doesn't matter which
		if (isov1 && isov2)
		{
			_compute(InteractionType::Halo, pv1, pv2, cl1, cl2, t, stream);
			return;
		}

		// One object vector. Compute just one interaction, with OV as the first argument
		if (isov1)
		{
			_compute(InteractionType::Halo, pv1, pv2, cl1, cl2, t, stream);
			return;
		}

		if (isov2)
		{
			_compute(InteractionType::Halo, pv2, pv1, cl2, cl1, t, stream);
			return;
		}

		// Both are particle vectors. Compute one interaction if pv1 == pv2 and two otherwise
		_compute(InteractionType::Halo, pv1, pv2, cl1, cl2, t, stream);
		if(pv1 != pv2)
			_compute(InteractionType::Halo, pv2, pv1, cl2, cl1, t, stream);
	}

	virtual ~Interaction() = default;
};
