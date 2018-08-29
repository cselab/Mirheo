#pragma once

#include <core/utils/pytypes.h>
#include "interface.h"

class FromArrayIC : public InitialConditions
{
private:
    PyContainer pos, vel;

public:
    FromArrayIC(const PyContainer &pos, const PyContainer &vel);

    void exec(const MPI_Comm& comm, ParticleVector *pv, DomainInfo domain, cudaStream_t stream) override;
};

