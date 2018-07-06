#pragma once

#include "interface.h"

#include <string>

/**
 * Initialize ParticleVector by reading the restart file.
 * \sa ParticleVector::checkpoint
 * \sa ParticleVector::restart
 */
class RestartIC : public InitialConditions
{
public:
	RestartIC(std::string path);

	void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override

	~RestartIC();
    
private:
	std::string path;
};
