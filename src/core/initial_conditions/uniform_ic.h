#pragma once

#include "interface.h"

/**
 * Initialize particles uniformly with the given density
 */
class UniformIC : public InitialConditions
{
private:
    float density;

public:
    UniformIC(float density);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~UniformIC();
};

