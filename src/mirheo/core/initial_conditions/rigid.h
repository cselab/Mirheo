#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

class RigidIC : public InitialConditions
{
public:
    RigidIC(const std::vector<ComQ>& com_q, const std::string& xyzfname);
    RigidIC(const std::vector<ComQ>& com_q, const std::vector<real3>& coords);
    RigidIC(const std::vector<ComQ>& com_q, const std::vector<real3>& coords,
            const std::vector<real3>& comVelocities);

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

    ~RigidIC();

private:
    std::vector<ComQ> com_q;
    std::vector<real3> coords;
    std::vector<real3> comVelocities;
};

} // namespace mirheo
