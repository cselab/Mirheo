#pragma once

#include <mpi.h>
#include <vector>

#include "core/domain.h"
#include "particle_vector.h"

namespace restart_helpers
{
    void exchangeParticles(const DomainInfo &domain, MPI_Comm comm, std::vector<Particle> &parts);
    void copyShiftCoordinates(const DomainInfo &domain, const std::vector<Particle> &parts, LocalParticleVector *local);
}
