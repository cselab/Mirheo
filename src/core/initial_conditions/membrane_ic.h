#pragma once

#include "interface.h"

#include <string>

/**
 * Initialize membranes.
 */
class MembraneIC : public InitialConditions
{
public:
    MembraneIC(std::string icfname, float globalScale = 1.0f);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~MembraneIC();

private:
    float globalScale;
    std::string icfname;
};
