#pragma once

#include <core/containers.h>
#include <core/field/from_function.h>
#include <core/utils/file_wrapper.h>

#include "interface.h"

class ParticleVector;

namespace VirialPressure
{
using ReductionType = double;
}

class VirialPressurePlugin : public SimulationPlugin
{
public:
    VirialPressurePlugin(const MirState *state, std::string name, std::string pvName,
                         FieldFunction func, float3 h, int dumpEvery);

    ~VirialPressurePlugin();

    void setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    void afterIntegration(cudaStream_t stream) override;
    void serializeAndSend(cudaStream_t stream) override;
    void handshake() override;

    bool needPostproc() override { return true; }

private:
    std::string pvName;
    int dumpEvery;
    bool needToSend = false;

    FieldFromFunction region;
    
    PinnedBuffer<VirialPressure::ReductionType> localVirialPressure {1};
    MirState::TimeType savedTime = 0;

    std::vector<char> sendBuffer;

    ParticleVector *pv;
};


class VirialPressureDumper : public PostprocessPlugin
{
public:
    VirialPressureDumper(std::string name, std::string path);
    
    void deserialize(MPI_Status& stat) override;
    void setup(const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void handshake() override;    

private:
    std::string path;

    bool activated = true;
    MPI_Datatype mpiReductionType;
    FileWrapper fdump;
};
