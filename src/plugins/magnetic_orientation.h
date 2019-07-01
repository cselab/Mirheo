#pragma once

#include <plugins/interface.h>
#include <string>
#include <functional>

#include <core/utils/folders.h>

class RigidObjectVector;

class MagneticOrientationPlugin : public SimulationPlugin
{
public:

    using UniformMagneticFunc = std::function<float3(float)>;
    
    MagneticOrientationPlugin(const MirState *state, std::string name, std::string rovName, float3 moment, UniformMagneticFunc magneticFunction);

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void beforeForces(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    std::string rovName;
    RigidObjectVector* rov;
    float3 moment;
    UniformMagneticFunc magneticFunction;
};

