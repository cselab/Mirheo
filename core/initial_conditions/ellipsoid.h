#pragma once

#include "interface.h"

#include <string>
#include <core/xml/pugixml.hpp>
#include <core/containers.h>

class EllipsoidIC : public InitialConditions
{
private:
	float separation;
	std::string xyzfname;

public:
	EllipsoidIC(float separation, std::string xyzfname) : separation(separation), xyzfname(xyzfname) {}

	void readXYZ(std::string fname, PinnedBuffer<float4>& positions);
	void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream) override;

	~EllipsoidIC() = default;
};
