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
    RigidIC(const std::vector<ComQ>& comQ, const std::string& xyzfname);
    RigidIC(const std::vector<ComQ>& comQ, const std::vector<real3>& coords);
    RigidIC(const std::vector<ComQ>& comQ, const std::vector<real3>& coords,
            const std::vector<real3>& comVelocities);

    ~RigidIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<ComQ> comQ_;
    std::vector<real3> coords_;
    std::vector<real3> comVelocities_;
};

} // namespace mirheo
