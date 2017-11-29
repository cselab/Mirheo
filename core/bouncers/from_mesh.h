#pragma once

#include "interface.h"

#include <core/containers.h>

/**
 * Implements bounce-back from mesh
 */
class BounceFromMesh : public Bouncer
{
public:
	BounceFromMesh(std::string name, float kbT = 0.5f) :
		Bouncer(name), nCollisions(1), kbT(kbT) {}

	~BounceFromMesh() = default;

protected:
	float kbT;
	static const int bouncesPerTri = 1;

	PinnedBuffer<int> nCollisions;
	DeviceBuffer<int2> collisionTable, tmp_collisionTable;
	DeviceBuffer<char> sortBuffer;

	void exec(ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream) override;
	void setup(ObjectVector* ov) override;
};
