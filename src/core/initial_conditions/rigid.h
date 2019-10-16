#pragma once

#include "interface.h"

#include <core/datatypes.h>

#include <vector>
#include <vector_types.h>

class RigidIC : public InitialConditions
{
public:
    RigidIC(const std::vector<ComQ>& com_q, const std::string& xyzfname);
    RigidIC(const std::vector<ComQ>& com_q, const std::vector<float3>& coords);
    RigidIC(const std::vector<ComQ>& com_q, const std::vector<float3>& coords,
            const std::vector<float3>& comVelocities);

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

    ~RigidIC();

private:
    std::vector<ComQ> com_q;
    std::vector<float3> coords;
    std::vector<float3> comVelocities;
};
