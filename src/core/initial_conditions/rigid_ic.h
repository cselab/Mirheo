#pragma once

#include "interface.h"

class RigidIC : public InitialConditions
{
private:
    std::string xyzfname, icfname;

public:
    RigidIC(std::string xyzfname, std::string icfname);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~RigidIC();
};
