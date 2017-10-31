#pragma once

#include "interface.h"

#include <core/containers.h>

class BounceFromMesh : public Bouncer
{
protected:
	static const int bouncePerTri = 1;

	PinnedBuffer<int> nCollisions;
	DeviceBuffer<int2> collisionTable, tmp_collisionTable;
	DeviceBuffer<char> sortBuffer;

	void exec(ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local) override;
	void setup(ObjectVector* ov) override { this->ov = ov; }

public:
	BounceFromMesh(std::string name) : Bouncer(name), nCollisions(1) {}

	~BounceFromMesh() = default;
};
