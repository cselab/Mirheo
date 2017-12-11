#pragma once
#include "interface.h"
#include <core/xml/pugixml.hpp>

/// Structure keeping all the parameters of the RBC model
struct RBCParameters
{
	float x0, p, ka, kb, kd, kv, gammaC, gammaT, kbT, mpow, theta, totArea0, totVolume0;
};

static const RBCParameters Lina_parameters =
{
		/*        x0 */ 0.457,
		/*         p */ 0.000906667,
		/*        ka */ 4900.0,
		/*        kb */ 44.4444,
		/*        kd */ 5000,
		/*        kv */ 7500.0,
		/*    gammaC */ 52.0,
		/*    gammaT */ 0.0,
		/*       kbT */ 0.0444,
		/*      mpow */ 2.0,
		/*     theta */ 6.97,
		/*   totArea */ 62.2242,
		/* totVolume */ 26.6649
};

/**
 * Implementation of RBC membrane forces
 */
class InteractionRBCMembrane : public Interaction
{
public:

	InteractionRBCMembrane(std::string name, RBCParameters parameters) :
		Interaction(name, 1.0f), parameters(parameters) {}

	void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;

	void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
	void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	~InteractionRBCMembrane() = default;

private:
	RBCParameters parameters;
};
