#pragma once

#include "interface.h"

#include <string>
#include <core/pvs/particle_vector.h>

/**
 * Initialize ParticleVector by reading the restart file.
 * \sa ParticleVector::checkpoint
 * \sa ParticleVector::restart
 */
class RestartIC : public InitialConditions
{
private:
	std::string path;

public:
	RestartIC(std::string path) : path(path) {};

	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override
	{
		pv->domain = domain;
		pv->restart(comm, path);
	}

	~RestartIC() = default;
};
