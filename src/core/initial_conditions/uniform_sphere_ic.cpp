#include "uniform_sphere_ic.h"
#include "helpers.h"

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
    float density;
    float3 center;
    float radius;
    bool inside;

UniformSphereIC::UniformSphereIC(float density, float3 center, float radius, bool inside) :
    density(density),
    center(center),
    radius(radius),
    inside(inside)
{}

UniformSphereIC::~UniformSphereIC() = default;
    
void UniformSphereIC::exec(const MPI_Comm& comm, ParticleVector* pv, cudaStream_t stream)
{
    auto filterOutParticles = [this](const DomainInfo& domain, const Particle& part) {
        float3 r = domain.local2global(part.r) - center;
        bool is_inside = length(r) <= radius;
        
        if (inside) return  is_inside;
        else        return !is_inside;
    };
    
    addUniformParticles(density, comm, pv, filterOutParticles, stream);
}

