#pragma once

#include <core/domain.h>
#include <core/datatypes.h>
#include <mpi.h>

class ParticleVector;

class StationaryWall_Box
{
public:
	StationaryWall_Box(float3 hi, float3 lo, bool inside) :
		hi(hi), lo(lo), inside(inside)
	{	}

	void setup(MPI_Comm& comm, DomainInfo domain) { this->domain = domain; }

	const StationaryWall_Box& handler() const { return *this; }

	__device__ __forceinline__ float operator()(float3 coo) const
	{
		float3 gr = domain.local2global(coo);

		float3 dist3;
		float3 d1 = gr - lo;
		float3 d2 = hi - gr;

		dist3.x = fabs(d1.x) < fabs(d2.x) ? d1.x : d2.x;
		dist3.y = fabs(d1.y) < fabs(d2.y) ? d1.y : d2.y;
		dist3.z = fabs(d1.z) < fabs(d2.z) ? d1.z : d2.z;

		float dist = dist3.x;
		if (fabs(dist) > fabs(dist3.y)) dist = dist3.y;
		if (fabs(dist) > fabs(dist3.z)) dist = dist3.z;

		return inside ? dist : -dist;
	}

private:
	float3 hi, lo;
	bool inside;

	DomainInfo domain;
};
