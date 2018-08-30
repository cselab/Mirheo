#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

class RigidIC : public InitialConditions
{
private:
    std::string xyzfname;
    PyTypes::VectorOfFloat7 com_q;

public:
    RigidIC(PyTypes::VectorOfFloat7 com_q, std::string xyzfname);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~RigidIC();
};
