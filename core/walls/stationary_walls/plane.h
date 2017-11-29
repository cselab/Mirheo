#pragma once

#include <core/domain.h>
#include <core/datatypes.h>
#include <mpi.h>

class ParticleVector;

class StationaryWall_Plane
{
public:
	StationaryWall_Plane(float3 normal, float3 pointThrough) :
		normal(normal), pointThrough(pointThrough)
	{
		normal = normalize(normal);
	}

	void setup(MPI_Comm& comm, DomainInfo domain) { this->domain = domain; }

	const StationaryWall_Plane& handler() const { return *this; }

	__device__ inline float operator()(float3 coo) const
	{
		float3 gr = domain.local2global(coo);
		float dist = dot(normal, gr - pointThrough);

		return dist;
	}

private:
	float3 normal, pointThrough;

	DomainInfo domain;
};
