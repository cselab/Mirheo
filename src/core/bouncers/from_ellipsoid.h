#pragma once

#include "interface.h"

/**
 * Implements bounce-back from the analytical ellipsoid shapes
 */
class BounceFromRigidEllipsoid : public Bouncer
{
public:

    BounceFromRigidEllipsoid(const YmrState *state, std::string name);
    ~BounceFromRigidEllipsoid();

    void setup(ObjectVector *ov) override;

protected:

    void exec(ParticleVector *pv, CellList *cl, bool local, cudaStream_t stream) override;
};
