#pragma once

#include "interface.h"
#include <mirheo/core/datatypes.h>

namespace mirheo
{

/**
 * Initialize particles uniformly with the given density
 */
class UniformIC : public InitialConditions
{
public:
    UniformIC(real numDensity);
    ~UniformIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    real numDensity_;
};


} // namespace mirheo
