#pragma once

#include "interface.h"

#include <string>
#include <core/pvs/particle_vector.h>

class RestartIC : public InitialConditions
{
private:
	std::string path;

public:
	RestartIC(std::string path) : path(path) {};

	void exec(const MPI_Comm& comm, ParticleVector* pv, float3 globalDomainStart, float3 localDomainSize, cudaStream_t stream) override
	{
		pv->globalDomainStart = globalDomainStart;
		pv->localDomainSize = localDomainSize;
		pv->restart(comm, path);
	}

	~RestartIC() = default;
};
