#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/pvs/packers/objects.h>

#include <string>

namespace mirheo
{

class ObjectVector;


class ObjectPortalCommon : public SimulationPlugin
{
protected:
    ObjectPortalCommon(const MirState *state, std::string name, std::string ovName,
                       real3 position, real3 size, int tag, MPI_Comm interCommExternal,
                       PackPredicate predicate);

public:
    ~ObjectPortalCommon();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;

    bool needPostproc() override { return false; }

protected:
    bool packPredicate(const DataManager::NamedChannelDesc &) noexcept;

    std::string ovName;
    ObjectVector *ov;
    std::string uuidChannelName;

    int tag;
    MPI_Comm interCommExternal;

    real3 localLo;  // Bounds of the local side of the portal
    real3 localHi;  // in the local coordinate system.
    ObjectPacker packer;
};


class ObjectPortalSource : public ObjectPortalCommon
{
public:
    ObjectPortalSource(const MirState *state, std::string name, std::string ovName, real3 src, real3 dst, real3 size, real4 plane, int tag, MPI_Comm interCommExternal);
    ~ObjectPortalSource();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

private:
    bool packPredicate(const DataManager::NamedChannelDesc &) noexcept;

    std::string oldSideChannelName;
    real4 plane;
    real3 shift;

    DeviceBuffer<real>   localOldSides;         // Per-object, previous value of ax+bx+cz+d for COM.
    DeviceBuffer<int64_t> uuidCounter {1};       // Global counter of UUIDs.

    DeviceBuffer<int>     outIdx;                // Local indices of the objects to send.
    PinnedBuffer<int64_t> outUUIDs;              // Their UUIDs.
    PinnedBuffer<char>    outBuffer;             // Outgoing buffer for packed objects.
    PinnedBuffer<int>     numObjectsToSend {1};  // How many objects to send.
};


class ObjectPortalDestination : public ObjectPortalCommon
{
public:
    ObjectPortalDestination(const MirState *state, std::string name, std::string ovName, real3 src, real3 dst, real3 size, int tag, MPI_Comm interCommExternal);
    ~ObjectPortalDestination();

    void setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm) override;
    void afterIntegration(cudaStream_t stream) override;

private:
    real3 shift;

    PinnedBuffer<int64_t> inUUIDs;
    PinnedBuffer<char> inBuffer;
    DeviceBuffer<bool> visited;
    DeviceBuffer<int> indexPairs;    // [2*k] = source index (inBuffer), [2*k+1] = destination (OV)
    PinnedBuffer<int> counters {2};  // [0] = #overwritten, [1] = #overwritten + #new.
};

} // namespace mirheo
