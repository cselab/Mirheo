#pragma once

#include "helpers.h"
#include "interface.h"

#include <vector_types.h>

namespace mirheo
{

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
class UniformFilteredIC : public InitialConditions
{
public:
    UniformFilteredIC(real density, PositionFilter filter);
    ~UniformFilteredIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    real density;
    PositionFilter filter;
};


} // namespace mirheo
