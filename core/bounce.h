#pragma once

#include <core/logger.h>
#include <core/containers.h>

class CellList;
class ParticleVector;
class ObjectVector;
class RigidObjectVector;

class Bouncer
{
protected:
	ParticleVector* pv;
	CellList* cl;

	ObjectVector* ov;
	float dt;

	virtual void exec(bool local, cudaStream_t stream) = 0;

public:

	Bouncer(float dt) : dt(dt) {};

	virtual void setup(ObjectVector* ov, ParticleVector* pv, CellList* cl)
	{
		this->ov = ov;
		this->pv = pv;
		this->cl = cl;
	}

	void bounceLocal(cudaStream_t stream) { exec(true,  stream); }
	void bounceHalo(cudaStream_t stream)  { exec(false, stream); }

	virtual ~Bouncer() = default;
};

class BounceFromRigidEllipsoid : public Bouncer
{
protected:
	void exec(bool local, cudaStream_t stream) override;
	RigidObjectVector* rov;

public:
	BounceFromRigidEllipsoid(float dt) : Bouncer(dt) {}

	void setup(ObjectVector* ov, ParticleVector* pv, CellList* cl) override;

	~BounceFromRigidEllipsoid() = default;
};


class BounceFromMesh : public Bouncer
{
protected:
	static const int bouncePerTri = 2;

	PinnedBuffer<int> nCollisions;
	DeviceBuffer<int2> collisionTable, tmp_collisionTable;
	DeviceBuffer<char> sortBuffer;

	void exec(bool local, cudaStream_t stream) override;


public:
	BounceFromMesh(float dt) : Bouncer(dt), nCollisions(1) {}

	~BounceFromMesh() = default;
};
