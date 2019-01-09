#pragma once

#include "interface.h"

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
class UniformSphereIC : public InitialConditions
{
private:
    float density;
    float3 center;
    float radius;
    bool inside;

public:
    UniformSphereIC(float density, float3 center, float radius, bool inside);
    ~UniformSphereIC();
    
    void exec(const MPI_Comm& comm, ParticleVector* pv, cudaStream_t stream) override;    
};

