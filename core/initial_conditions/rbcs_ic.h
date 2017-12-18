#pragma once

#include "interface.h"

#include <string>
#include <core/xml/pugixml.hpp>
#include <core/containers.h>

/**
 * Initialize red blood cells.
 */
class RBC_IC : public InitialConditions
{
public:
	RBC_IC(std::string icfname) : icfname(icfname)
	{	}

	void readVertices(std::string fname, PinnedBuffer<float4>& positions);
	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~RBC_IC() = default;

private:
	std::string icfname;
};
