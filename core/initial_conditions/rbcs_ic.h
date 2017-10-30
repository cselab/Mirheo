#pragma once

#include "interface.h"

#include <string>
#include <core/xml/pugixml.hpp>
#include <core/containers.h>

class RBC_IC : public InitialConditions
{
private:
	std::string xyzfname, icfname;

public:
	RBC_IC(std::string xyzfname, std::string icfname) : icfname(icfname), xyzfname(xyzfname) {}

	void readVertices(std::string fname, PinnedBuffer<float4>& positions, cudaStream_t stream);
	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~RBC_IC() = default;
};
