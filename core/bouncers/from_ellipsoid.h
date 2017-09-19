#pragma once

#include "interface.h"

class BounceFromRigidEllipsoid : public Bouncer
{
protected:
	void exec(ObjectVector* ov, ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local) override;

public:
	BounceFromRigidEllipsoid(std::string name) : Bouncer(name) {}

	~BounceFromRigidEllipsoid() = default;
};
