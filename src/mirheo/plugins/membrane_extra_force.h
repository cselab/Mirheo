#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/plugins/interface.h>

#include <array>
#include <string>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

class MembraneVector;

class MembraneExtraForcePlugin : public SimulationPlugin
{
public:

    MembraneExtraForcePlugin(const MirState *state, std::string name, std::string pvName, const std::vector<real3>& forces);

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string pvName;
    MembraneVector *pv;
    DeviceBuffer<Force> forces;
};

} // namespace mirheo
