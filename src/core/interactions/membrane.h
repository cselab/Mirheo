#pragma once
#include "interface.h"
#include <functional>

/// Structure keeping all the parameters of the RBC model
struct MembraneParameters
{
    float x0, p, ka, kb, kd, kv, gammaC, gammaT, kbT, mpow, theta, totArea0, totVolume0;
};

static const MembraneParameters Lina_parameters =
{
        /*        x0 */ 0.457,
        /*         p */ 0.000906667 * 1.5,
        /*        ka */ 4900.0,
        /*        kb */ 44.4444 * 1.5*1.5,
        /*        kd */ 5000,
        /*        kv */ 7500.0,
        /*    gammaC */ 52.0 * 1.5,
        /*    gammaT */ 0.0,
        /*       kbT */ 0.0444 * 1.5*1.5,
        /*      mpow */ 2.0,
        /*     theta */ 6.97,
        /*   totArea */ 62.2242 * 1.5*1.5,
        /* totVolume */ 26.6649 * 1.5*1.5*1.5
};

/**
 * Implementation of RBC membrane forces
 */
class InteractionMembrane : public Interaction
{
public:

    InteractionMembrane(std::string name, MembraneParameters parameters, bool stressFree, float growUntil);

    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;

    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

    ~InteractionMembrane();

private:

    bool stressFree;
    std::function< float(float) > scaleFromTime;
    MembraneParameters parameters;
};
