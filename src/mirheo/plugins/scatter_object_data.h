#pragma once

#include "interface.h"

namespace mirheo
{

class ObjectVector;

class ScatterObjectDataPlugin : public SimulationPlugin
{
public:
    ScatterObjectDataPlugin(const MirState *state, std::string name, std::string ovName,
                            std::string channelName, std::string savedName);

    void beforeForces(cudaStream_t stream) override;
    
    bool needPostproc() override {return false;}

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

private:
    std::string ovName;
    ObjectVector *ov;
    std::string channelName, savedName;
};

} // namespace mirheo
