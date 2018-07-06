#pragma once

#include "interface.h"

#include <string>

/**
 * Initialize membranes.
 */
class Membrane_IC : public InitialConditions
{
public:
	Membrane_IC(std::string icfname, float globalScale = 1.0f);

	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

	~Membrane_IC();

private:
	float globalScale;
	std::string icfname;
};
