#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/pvs/packers/particles.h>

#include <string>

class ParticleVector;


class ParticlePortalCommon : public SimulationPlugin
{
protected:
    ParticlePortalCommon(const MirState *state, std::string name, std::string pvName, float3 position, float3 size, int tag, MPI_Comm interCommExternal);

public:
    bool needPostproc() override { return false; }

    ~ParticlePortalCommon();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

protected:
    std::string pvName;
    ParticleVector *pv;

    int tag;
    MPI_Comm interCommExternal;

    float3 localLo;  // Bounds of the local side of the portal.
    float3 localHi;
    ParticlePacker packer;
};


class ParticlePortalSource : public ParticlePortalCommon
{
public:
    ParticlePortalSource(const MirState *state, std::string name, std::string pvName, float3 src, float3 dst, float3 size, int tag, MPI_Comm interCommExternal);

    ~ParticlePortalSource();

    void beforeCellLists(cudaStream_t stream) override;

    bool needPostproc() override { return false; }

private:
    float3 shift;

    PinnedBuffer<int>  numParticlesToSend {2};  // Two counters for two phases.
    PinnedBuffer<char> outBuffer;
};


class ParticlePortalDestination : public ParticlePortalCommon
{
public:
    // Using the same interface as for portal source, even though we don't need
    // `src` here.
    ParticlePortalDestination(const MirState *state, std::string name, std::string pvName, float3 src, float3 dst, float3 size, int tag, MPI_Comm interCommExternal);

    ~ParticlePortalDestination();

    void beforeCellLists(cudaStream_t stream) override;

private:
    float3 shift;

    PinnedBuffer<char> inBuffer;
};
