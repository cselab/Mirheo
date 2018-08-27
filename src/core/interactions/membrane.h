#pragma once
#include "interface.h"
#include <functional>

/// Structure keeping all the parameters of the RBC model
struct MembraneParameters
{
    float x0, ks, ka, kb, kd, kv, gammaC, gammaT, kbT, mpow, theta, totArea0, totVolume0;

    bool fluctuationForces;
    float dt;
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
