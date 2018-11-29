#pragma once

#include "membrane.h"

struct JuelicherBendingParameters
{
    float kb, C0, kad, DA0;
};

class InteractionMembraneJuelicher : public InteractionMembrane
{
public:

    InteractionMembraneJuelicher(std::string name, MembraneParameters parameters, JuelicherBendingParameters bendingParameters, bool stressFree, float growUntil);
    ~InteractionMembraneJuelicher();
    
    void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2) override;

protected:

    JuelicherBendingParameters bendingParameters;

    void bendingForces(float scale, MembraneVector *ov, MembraneMeshView mesh, cudaStream_t stream) override;
};
