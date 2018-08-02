#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

#include <string>

/**
 * Initialize membranes.
 */
class MembraneIC : public InitialConditions
{
public:
    MembraneIC(ICvector com_q, float globalScale = 1.0f);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~MembraneIC();

private:
    float globalScale;
    ICvector com_q;
};
