#pragma once

#include "interface.h"

#include <string>
#include <core/xml/pugixml.hpp>
#include <core/containers.h>

/**
 * Initialize red blood cells.
 */
class Membrane_IC : public InitialConditions
{
public:
	Membrane_IC(std::string icfname, float globalScale = 1.0f) : icfname(icfname), globalScale(globalScale)
	{	}

	void readVertices(std::string fname, PinnedBuffer<float4>& positions);
	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~Membrane_IC() = default;

private:
	float globalScale;
	std::string icfname;
};
