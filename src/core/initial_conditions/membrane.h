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
    MembraneIC(PyTypes::VectorOfFloat7 com_q, float globalScale = 1.0f);

    void exec(const MPI_Comm& comm, ParticleVector* pv, cudaStream_t stream) override;

    ~MembraneIC();

private:
    PyTypes::VectorOfFloat7 com_q;
    float globalScale;
};
