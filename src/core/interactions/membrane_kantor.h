#pragma once

#include "membrane.h"

struct KantorBendingParameters
{
    float kb, theta;
};

class InteractionMembraneKantor : public InteractionMembrane
{
public:

    InteractionMembraneKantor(const YmrState *state, std::string name, MembraneParameters parameters,
                              KantorBendingParameters bendingParameters, bool stressFree, float growUntil);
    ~InteractionMembraneKantor();

protected:

    KantorBendingParameters bendingParameters;

    void bendingForces(float scale, MembraneVector *ov, MembraneMeshView mesh, cudaStream_t stream) override;
};
