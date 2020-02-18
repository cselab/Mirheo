#include "uniform_filtered.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

UniformFilteredIC::UniformFilteredIC(real numDensity, PositionFilter filter) :
    numDensity_(numDensity),
    filter_(filter)
{}

UniformFilteredIC::~UniformFilteredIC() = default;
    
void UniformFilteredIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    setUniformParticles(numDensity_, comm, pv, filter_, stream);
}


} // namespace mirheo
