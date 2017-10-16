#pragma once

#include <core/datatypes.h>
#include <mpi.h>

class ParticleVector;

class StationaryWall_Sphere
{
public:
	StationaryWall_Sphere(float3 center, float radius, bool inside) :
		center(center), radius(radius), inside(inside)
	{	}

	void setup(MPI_Comm& comm, float3 globalDomainSize, float3 globalDomainStart, float3 localDomainSize) {}

	const StationaryWall_Sphere& handler() const { return *this; }

	__device__ __forceinline__ float operator()(const PVview view, float3 coo) const
	{
		float3 gr = view.local2global(coo);
		float dist = sqrtf(dot(gr-center, gr-center));

		return inside ? dist - radius : radius - dist;
	}

private:
	float3 center;
	float radius;

	bool inside;
};
