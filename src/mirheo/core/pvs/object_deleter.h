#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/packers/objects.h>

namespace mirheo
{

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

    /// Copy the channels of the given LocalObjectVector, allocate and reset marks.
    void update(LocalObjectVector *lov, cudaStream_t stream);

    /// Create a handler object to be used from the device code.
    ObjectDeleterHandler handler();

    /// Delete marked objects and (WIP) optionally copy their particles to a particle vector.
    void deleteObjects(LocalObjectVector *lov, cudaStream_t stream, LocalParticleVector *lpvTarget = nullptr);

private:
    DeviceBuffer<bool> marks;
    DeviceBuffer<int> prefixSum;
    DeviceBuffer<char> scanBuffer;
    HostBuffer<int> numRemoved {1};
    std::unique_ptr<LocalObjectVector> tmpLov;
    ObjectPacker packerSrc;
    ObjectPacker packerDst;
    ParticlePacker partPackerDst;
};

} // namespace mirheo
