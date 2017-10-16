#pragma once

#include "interface.h"

class BounceFromRigidEllipsoid : public Bouncer
{
protected:
	void exec(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local) override;

public:
	BounceFromRigidEllipsoid(std::string name) : Bouncer(name) {}

	void setup(ObjectVector* ov) override { this->ov = ov; }

	~BounceFromRigidEllipsoid() = default;
};
