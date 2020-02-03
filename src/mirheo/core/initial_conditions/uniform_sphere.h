#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <vector_types.h>

namespace mirheo
{

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
class UniformSphereIC : public InitialConditions
{
public:
    UniformSphereIC(real numDensity, real3 center, real radius, bool inside);
    ~UniformSphereIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    real numDensity_;
    real3 center_;
    real  radius_;
    bool inside_;
};

} // namespace mirheo
