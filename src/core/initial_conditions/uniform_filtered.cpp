#include "uniform_filtered.h"

#include <core/pvs/particle_vector.h>

UniformFilteredIC::UniformFilteredIC(float density, PositionFilter filter) :
    density(density),
    filter(filter)
{}

UniformFilteredIC::~UniformFilteredIC() = default;
    
void UniformFilteredIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    addUniformParticles(density, comm, pv, filter, stream);
}

