#pragma once

#include <core/logger.h>
#include <core/containers.h>

class CellList;
class ParticleVector;
class ObjectVector;
class RigidObjectVector;

class Bounce
{
protected:
	ParticleVector* pv;
	CellList* cl;

	ObjectVector* ov;
	float dt;

	virtual void exec(bool local, cudaStream_t stream) = 0;

public:

	Bounce(float dt) : dt(dt) {};

	virtual void setup(ObjectVector* ov, ParticleVector* pv, CellList* cl)
	{
		this->ov = ov;
		this->pv = pv;
		this->cl = cl;
	}

	void bounceLocal(cudaStream_t stream) { exec(true,  stream); }
	void bounceHalo(cudaStream_t stream)  { exec(false, stream); }

	virtual ~Bounce() = default;
};

class BounceFromRigidEllipsoid : public Bounce
{
protected:
	void exec(bool local, cudaStream_t stream) override;
	RigidObjectVector* rov;

public:
	BounceFromRigidEllipsoid(float dt) : Bounce(dt) {}

	void setup(ObjectVector* ov, ParticleVector* pv, CellList* cl) override;

	~BounceFromRigidEllipsoid() = default;
};


class BounceFromMesh : public Bounce
{
protected:
	static const int bouncePerTri = 2;

	PinnedBuffer<int> nCollisions;
	DeviceBuffer<int2> collisionTable, tmp_collisionTable;
	DeviceBuffer<char> sortBuffer;

	void exec(bool local, cudaStream_t stream) override;


public:
	BounceFromMesh(float dt) : Bounce(dt), nCollisions(1) {}

	~BounceFromMesh() = default;
};
