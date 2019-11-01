#pragma once

#include "interface.h"

#include <vector_types.h>

namespace mirheo
{

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
class UniformSphereIC : public InitialConditions
{
public:
    UniformSphereIC(real density, real3 center, real radius, bool inside);
    ~UniformSphereIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    real density;
    real3 center;
    real  radius;
    bool inside;
};

} // namespace mirheo
