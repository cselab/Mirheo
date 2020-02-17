#include "uniform_sphere.h"
#include "helpers.h"

#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

UniformSphereIC::UniformSphereIC(real numDensity, real3 center, real radius, bool inside) :
    numDensity_(numDensity),
    center_(center),
    radius_(radius),
    inside_(inside)
{}

UniformSphereIC::~UniformSphereIC() = default;
    
void UniformSphereIC::exec(const MPI_Comm& comm, ParticleVector* pv, cudaStream_t stream)
{
    auto filterSphere = [this](real3 r) {
        r -= center_;
        const bool is_inside = length(r) <= radius_;
        
        if (inside_) return  is_inside;
        else         return !is_inside;
    };
    
    setUniformParticles(numDensity_, comm, pv, filterSphere, stream);
}

} // namespace mirheo
