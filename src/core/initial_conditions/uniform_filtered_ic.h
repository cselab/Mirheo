#pragma once

#include "helpers.h"
#include "interface.h"

#include <core/utils/pytypes.h>

/**
 * Initialize particles uniformly inside or outside a sphere with the given density
 */
class UniformFilteredIC : public InitialConditions
{
private:
    float density;
    PositionFilter filter;

public:
    UniformFilteredIC(float density, PositionFilter filter);
    UniformFilteredIC(float density, std::function<bool(PyTypes::float3)> filter);
    ~UniformFilteredIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    
};

