#pragma once

#include "interface.h"

#include <string>
#include <core/xml/pugixml.hpp>
#include <core/containers.h>

class EllipsoidIC : public InitialConditions
{
private:
	std::string xyzfname, icfname;

public:
	EllipsoidIC(std::string xyzfname, std::string icfname) : icfname(icfname), xyzfname(xyzfname) {}

	void readXYZ(std::string fname, PinnedBuffer<float4>& positions, cudaStream_t stream);
	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~EllipsoidIC() = default;
};
