#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/pvs/packers/particles.h>

#include <string>

namespace mirheo
{

class ParticleVector;

class ParticlePortalCommon : public SimulationPlugin
{
protected:
    ParticlePortalCommon(const MirState *state, std::string name, std::string pvName, real3 position, real3 size, int tag, MPI_Comm interCommExternal);

public:
    ~ParticlePortalCommon();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:
    std::string pvName;
    ParticleVector *pv;

    int tag;
    MPI_Comm interCommExternal;

    real3 localLo;  // Bounds of the local side of the portal.
    real3 localHi;
    ParticlePacker packer;
};


class ParticlePortalSource : public ParticlePortalCommon
{
public:
    ParticlePortalSource(const MirState *state, std::string name, std::string pvName, real3 src, real3 dst, real3 size, int tag, MPI_Comm interCommExternal);
    ~ParticlePortalSource();

    void beforeCellLists(cudaStream_t stream) override;

private:
    real3 shift;

    PinnedBuffer<int>  numParticlesToSend {2};  // Two counters for two phases.
    PinnedBuffer<char> outBuffer;
};


class ParticlePortalDestination : public ParticlePortalCommon
{
public:
    // Using the same interface as for portal source, even though we don't need
    // `src` here.
    ParticlePortalDestination(const MirState *state, std::string name, std::string pvName, real3 src, real3 dst, real3 size, int tag, MPI_Comm interCommExternal);
    ~ParticlePortalDestination();

    void beforeCellLists(cudaStream_t stream) override;

private:
    real3 shift;

    PinnedBuffer<char> inBuffer;
};

} // namespace mirheo
