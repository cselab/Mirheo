#pragma once

#include "interface.h"

/**
 * Initialize particles uniformly with the given density
 */
class UniformIC : public InitialConditions
{
private:
    real density;

public:
    UniformIC(real density);

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

    ~UniformIC();
};

