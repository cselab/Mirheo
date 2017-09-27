#pragma once

#include "rigid_object_vector.h"

class LocalRigidEllipsoidObjectVector: public LocalRigidObjectVector
{
public:

	LocalRigidEllipsoidObjectVector(const int objSize, const int nObjects = 0, cudaStream_t stream = 0) :
		LocalRigidObjectVector(objSize, nObjects)
	{ }

	virtual ~LocalRigidEllipsoidObjectVector() = default;
};

class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
	float3 axes;

	RigidEllipsoidObjectVector(std::string name, float mass, const int objSize, float3 axes, const int nObjects = 0) :
		RigidObjectVector(name, mass, objSize, nObjects), axes(axes)
	{}

	float3 getInertiaTensor() override;

	LocalRigidEllipsoidObjectVector* local() { return static_cast<LocalRigidEllipsoidObjectVector*>(_local); }
	LocalRigidEllipsoidObjectVector* halo()  { return static_cast<LocalRigidEllipsoidObjectVector*>(_halo);  }

	virtual ~RigidEllipsoidObjectVector() {};
};


/**
 * GPU-compatibe struct of all the relevant data
 */
struct REOVview : public ROVview
{
	float3 axes, invAxes;


	REOVview(RigidEllipsoidObjectVector* ov, LocalRigidEllipsoidObjectVector* lov) :
		ROVview(static_cast<RigidObjectVector*>(ov), static_cast<LocalRigidObjectVector*>(lov))
	{
		axes = ov->axes;
		invAxes = 1.0 / axes;
	}
};


