#pragma once

#include <core/containers.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/packers/objects.h>

class ObjectVector;

class ObjectDeleterHandler
{
public:
    inline __D__ void mark(int objectIndex)
    {
        devPtr[objectIndex] = 1;
    }

private:
    friend class ObjectDeleter;
    ObjectDeleterHandler(bool *devPtr) : devPtr(devPtr) {}

    bool *devPtr;
};

class ObjectDeleter
{
public:
    ObjectDeleter();
    ~ObjectDeleter();

    /// Copy the structure of the initial LocalObjectVector, allocate and reset marks.
    void update(LocalObjectVector *lov, cudaStream_t stream);

    /// Create a handler object to be used from the device code.
    ObjectDeleterHandler handler();

    /// Delete marked objects and optionally copy their particles to a particle vector.
    void deleteObjects(LocalObjectVector *lov, cudaStream_t stream, LocalParticleVector *lpvTarget = nullptr);

private:
    DeviceBuffer<bool> marks;  // Could be bool, but wouldn't work with cub::ExclusiveScan.
    DeviceBuffer<int> prefixSum;
    DeviceBuffer<char> scanBuffer;
    HostBuffer<int> numRemoved {1};
    std::unique_ptr<LocalObjectVector> tmpLov;
    ObjectPacker packerSrc;
    ObjectPacker packerDst;
    ParticlePacker partPackerDst;
};
