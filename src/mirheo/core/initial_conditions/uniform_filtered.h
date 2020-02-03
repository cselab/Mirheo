#pragma once

#include "helpers.h"
#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <vector_types.h>

namespace mirheo
{

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
class UniformFilteredIC : public InitialConditions
{
public:
    UniformFilteredIC(real numDensity, PositionFilter filter);
    ~UniformFilteredIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    real numDensity_;
    PositionFilter filter_;
};


} // namespace mirheo
