#include "uniform_filtered.h"

#include <core/pvs/particle_vector.h>

UniformFilteredIC::UniformFilteredIC(float density, PositionFilter filter) :
    density(density),
    filter(filter)
{}

UniformFilteredIC::UniformFilteredIC(float density, std::function<bool(PyTypes::float3)> pyfilter) :
    UniformFilteredIC(density,
                      [pyfilter](float3 r) {
                          PyTypes::float3 pyr {r.x, r.y, r.z};
                          return pyfilter(pyr);
                      })
{}

UniformFilteredIC::~UniformFilteredIC() = default;
    
void UniformFilteredIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    addUniformParticles(density, comm, pv, filter, stream);
}

