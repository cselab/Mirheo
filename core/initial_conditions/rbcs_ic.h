#pragma once

#include "interface.h"

#include <string>
#include <core/xml/pugixml.hpp>
#include <core/containers.h>

class RBC_IC : public InitialConditions
{
private:
	std::string offfname, icfname;

public:
	RBC_IC(std::string offfname, std::string icfname) : icfname(icfname), offfname(offfname) {}

	void readVertices(std::string fname, PinnedBuffer<float4>& positions);
	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~RBC_IC() = default;
};
