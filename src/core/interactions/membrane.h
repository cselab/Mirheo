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
    float dt;
};

/**
 * Implementation of RBC membrane forces
 */
class InteractionMembrane : public Interaction
{
public:

    InteractionMembrane(std::string name, MembraneParameters parameters, bool stressFree, float growUntil);
    ~InteractionMembrane();
    
    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;

    void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
    void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;    

protected:

    bool stressFree;
    std::function< float(float) > scaleFromTime;
    MembraneParameters parameters;

    virtual void bendingForces(float scale, MembraneVector *ov, MembraneMeshView mesh, cudaStream_t stream) = 0;
};
