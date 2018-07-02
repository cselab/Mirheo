#pragma once

#include "interface.h"

/**
 * Implements bounce-back from the analytical ellipsoid shapes
 */
class BounceFromRigidEllipsoid : public Bouncer
{
protected:
	void exec(ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream) override;

public:

	BounceFromRigidEllipsoid(std::string name);

	void setup(ObjectVector* ov) override;

	~BounceFromRigidEllipsoid() = default;
};
