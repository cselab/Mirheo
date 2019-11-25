#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/plugins.h>
#include <mirheo/core/pvs/object_deleter.h>

namespace mirheo
{

class ObjectToParticlesPlugin : public SimulationPlugin
{
public:
    ObjectToParticlesPlugin(const MirState *state, std::string name, std::string ovName, std::string pvName, real4 plane);
    ~ObjectToParticlesPlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

protected:
    std::string ovName;
    std::string pvName;
    ObjectVector   *ov;  // From.
    ParticleVector *pv;  // To.

    ObjectDeleter deleter;
    real4 plane;  // Local coordinate system.
};

} // namespace mirheo
