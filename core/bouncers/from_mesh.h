#pragma once

#include "interface.h"

class BounceFromMesh : public Bouncer
{
protected:
	static const int bouncePerTri = 2;

	PinnedBuffer<int> nCollisions;
	DeviceBuffer<int2> collisionTable, tmp_collisionTable;
	DeviceBuffer<char> sortBuffer;

	void exec(ObjectVector* ov, ParticleVector* pv, CellList* cl, float dt, cudaStream_t stream, bool local) override;

public:
	BounceFromMesh(std::string name) : Bouncer(name), nCollisions(1) {}

	~BounceFromMesh() = default;
};
