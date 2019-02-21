#pragma once

#include "interface.h"

#include <functional>

class MembraneMeshView;
class MembraneVector;

/// Structure keeping elastic parameters of the RBC model
struct MembraneParameters
{
    float x0, ks, ka, kd, kv, gammaC, gammaT, kbT, mpow, totArea0, totVolume0;
    bool fluctuationForces;
};

/**
 * Implementation of RBC membrane forces
 */
template <class DihedralInteraction>
class InteractionMembrane : public Interaction
{
public:

    InteractionMembrane(const YmrState *state, std::string name, MembraneParameters parameters,
                        DihedralInteraction dihedralInteraction, bool stressFree, float growUntil);
    ~InteractionMembrane();
    
    void setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2) override;

    void local (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;
    void halo  (ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2, cudaStream_t stream) override;

protected:

    bool stressFree;
    std::function< float(float) > scaleFromTime;
    MembraneParameters parameters;
    DihedralInteraction dihedralInteraction;
};
