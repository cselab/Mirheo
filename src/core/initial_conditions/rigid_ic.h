#pragma once

#include "interface.h"

#include <core/utils/pytypes.h>

class RigidIC : public InitialConditions
{
private:
    PyTypes::VectorOfFloat3 coords;
    PyTypes::VectorOfFloat7 com_q;
    PyTypes::VectorOfFloat3 comVelocities;

public:
    RigidIC(PyTypes::VectorOfFloat7 com_q, std::string xyzfname);
    RigidIC(PyTypes::VectorOfFloat7 com_q, const PyTypes::VectorOfFloat3& coords);
    RigidIC(PyTypes::VectorOfFloat7 com_q, const PyTypes::VectorOfFloat3& coords, const PyTypes::VectorOfFloat3& comVelocities);

    void exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream) override;

    ~RigidIC();
};
