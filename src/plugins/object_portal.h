#pragma once

#include "interface.h"

#include <core/containers.h>
#include <core/pvs/packers/objects.h>

#include <string>

class ObjectVector;


class ObjectPortalCommon : public SimulationPlugin
{
protected:
    ObjectPortalCommon(const MirState *state, std::string name, std::string ovName, float3 position, float3 size, int tag, MPI_Comm interCommExternal);

public:
    bool needPostproc() override { return false; }

    ~ObjectPortalCommon();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

protected:
    std::string ovName;
    ObjectVector *ov;
    std::string uuidChannelName;

    int tag;
    MPI_Comm interCommExternal;

    float3 localLo;  // Bounds of the local side of the portal.
    float3 localHi;
    ObjectPacker packer;
};

class ObjectPortalSource : public ObjectPortalCommon
{
public:
    ObjectPortalSource(const MirState *state, std::string name, std::string ovName, float3 src, float3 dst, float3 size, float4 plane, int tag, MPI_Comm interCommExternal);

    ~ObjectPortalSource();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

private:
    std::string oldSideChannelName;
    float4 plane;
    float3 shift;

    DeviceBuffer<float>   localOldSides;         // Per-object, previous value of ax+bx+cz+d for COM.
    DeviceBuffer<int64_t> uuidCounter {1};       // Global counter of UUIDs.

    DeviceBuffer<int>     outIdx;                // Local indices of the objects to send.
    PinnedBuffer<int64_t> outUUIDs;              // Their UUIDs.
    PinnedBuffer<char>    outBuffer;             // Outgoing buffer for packed objects.
    PinnedBuffer<int>     numObjectsToSend {1};  // How many objects to send.
};


class ObjectPortalDestination : public ObjectPortalCommon
{
public:
    ObjectPortalDestination(const MirState *state, std::string name, std::string ovName, float3 src, float3 dst, float3 size, int tag, MPI_Comm interCommExternal);

    ~ObjectPortalDestination();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

private:
    float3 shift;

    PinnedBuffer<int64_t> inUUIDs;
    PinnedBuffer<char> inBuffer;
    DeviceBuffer<bool> visited;
    DeviceBuffer<int> indexPairs;    // [2*k] = source index (inBuffer), [2*k+1] = destination (OV)
    PinnedBuffer<int> counters {2};  // [0] = #overwritten, [1] = #overwritten + #new.
};

