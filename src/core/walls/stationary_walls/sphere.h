#pragma once

#include <core/domain.h>
#include <core/datatypes.h>
#include <mpi.h>

class ParticleVector;

class StationaryWall_Sphere
{
public:
	StationaryWall_Sphere(float3 center, float radius, bool inside) :
		center(center), radius(radius), inside(inside)
	{	}

	void setup(MPI_Comm& comm, 	DomainInfo domain) { this->domain = domain; }

	const StationaryWall_Sphere& handler() const { return *this; }

	__device__ inline float operator()(float3 coo) const
	{
		float3 gr = domain.local2global(coo);
		float dist = sqrtf(dot(gr-center, gr-center));

		return inside ? dist - radius : radius - dist;
	}

private:
	float3 center;
	float radius;

	bool inside;

	DomainInfo domain;
};
