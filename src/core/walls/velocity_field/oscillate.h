#pragma once

#include <core/domain.h>
#include <core/datatypes.h>
#include <mpi.h>

class ParticleVector;

class VelocityField_Oscillate
{
public:
	VelocityField_Oscillate(float3 vel, int period) :
		vel(vel), period(period)
	{
		if (period <= 0)
			die("Oscillating period should be strictly positive");
	}

	void setup(MPI_Comm& comm, DomainInfo domain)
	{
		cosOmega = cos(2*M_PI * (float)count / period);
		count++;
	}

	const VelocityField_Oscillate& handler() const { return *this; }

	__device__ inline float3 operator()(float3 coo) const
	{
		return vel * cosOmega;
	}

private:
	float3 vel;
	int period;
	int count{0};

	float cosOmega;

	DomainInfo domain;
};
