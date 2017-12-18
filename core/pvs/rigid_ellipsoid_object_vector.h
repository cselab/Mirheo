#pragma once

#include "rigid_object_vector.h"

class RigidEllipsoidObjectVector : public RigidObjectVector
{
public:
	float3 axes;

	RigidEllipsoidObjectVector(std::string name, float mass, const int objSize, float3 axes, const int nObjects = 0) :
		RigidObjectVector(
				name, mass,
				mass*objSize / 5.0f * make_float3(
						sqr(axes.y) + sqr(axes.z),
						sqr(axes.z) + sqr(axes.x),
						sqr(axes.x) + sqr(axes.y) ),
				objSize,
				Mesh(),
				nObjects),
		axes(axes)
	{	}

	virtual ~RigidEllipsoidObjectVector() {};
};

#include "views/reov.h"

