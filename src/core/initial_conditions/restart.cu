#pragma once

#include "restart.h"
#include <core/pvs/particle_vector.h>

RestartIC::RestartIC(std::string path) : path(path)
{   }

void RestartIC::exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override
{
    pv->domain = domain;
    pv->restart(comm, path);
}

RestartIC::~RestartIC() = default;
