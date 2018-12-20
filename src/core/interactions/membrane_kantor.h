#pragma once

#include "membrane.h"

struct KantorBendingParameters
{
    float kb, theta;
};

class InteractionMembraneKantor : public InteractionMembrane
{
public:

    InteractionMembraneKantor(std::string name, const YmrState *state, MembraneParameters parameters,
                              KantorBendingParameters bendingParameters, bool stressFree, float growUntil);
    ~InteractionMembraneKantor();

protected:

    KantorBendingParameters bendingParameters;

    void bendingForces(float scale, MembraneVector *ov, MembraneMeshView mesh, cudaStream_t stream) override;
};
