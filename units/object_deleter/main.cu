#include "../packers/common.h"

#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/pvs/object_deleter.cu>

#include <gtest/gtest.h>

using namespace mirheo;

namespace mirheo { Logger logger; }

bool verbose = false;

__global__ void setIds(int64_t *ids, int N, int factor)
{
    int oid = blockIdx.x * blockDim.x + threadIdx.x;
    if (oid >= N) return;
    ids[N] = oid * oid + factor * oid;  // Quasirandom value.
}

__global__ void markParticles(ObjectDeleterHandler handler, int *mask, int numObjects)
{
    int oid = blockIdx.x * blockDim.x + threadIdx.x;
    if (oid >= numObjects) return;

    if (mask[oid])
        handler.mark(oid);
}

struct ObjectData
{
    ObjectData(LocalObjectVector *lov, cudaStream_t stream) {
        partPositions .copy(lov->positions(), stream);
        partVelocities.copy(lov->velocities(), stream);
        partIds       .copy(*lov->dataPerParticle.getData<int64_t>(ChannelNames::globalIds), stream);
        objIds        .copy(*lov->dataPerObject.getData<int64_t>(ChannelNames::globalIds), stream);
    }

    HostBuffer<float4> partPositions;
    HostBuffer<float4> partVelocities;
    HostBuffer<int64_t> partIds;
    HostBuffer<int64_t> objIds;
};

bool operator==(float4 a, float4 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

void compareObjects(const ObjectData &a, const ObjectData &b, int aId, int bId)
{
    int objSize = a.partIds.size() / a.objIds.size();
    assert(aId < a.partIds.size());
    assert(bId < b.partIds.size());

    ASSERT_EQ(a.objIds[aId], b.objIds[bId]);

    for (int i = 0; i < objSize; ++i) {
        int aPid = aId * objSize + i;
        int bPid = bId * objSize + i;
        ASSERT_EQ(a.partPositions[aPid],  b.partPositions[bPid]);
        ASSERT_EQ(a.partVelocities[aPid], b.partVelocities[bPid]);
        ASSERT_EQ(a.partIds[aPid],        b.partIds[bPid]);
    }
}


void test_delete(const std::vector<int> &mask, const int objSize)
{
    const int nthreads = 128;
    const float3 length{100.f, 100.f, 100.f};
    const DomainInfo domain{length, {0, 0, 0}, length};
    const float dt = 0;  // dummy
    MirState state(domain, dt);

    const int numObjects = (int)mask.size();

    std::unique_ptr<ObjectVector> ov = initializeRandomREV(MPI_COMM_WORLD, &state, numObjects, objSize);

    ov->requireDataPerObject  <int64_t>(ChannelNames::globalIds, DataManager::PersistenceMode::None);
    ov->requireDataPerParticle<int64_t>(ChannelNames::globalIds, DataManager::PersistenceMode::None);
    int64_t *partIds = ov->local()->dataPerParticle.getData<int64_t>(ChannelNames::globalIds)->devPtr();
    int64_t *objIds  = ov->local()->dataPerObject  .getData<int64_t>(ChannelNames::globalIds)->devPtr();

    SAFE_KERNEL_LAUNCH(setIds,
                       getNblocks(numObjects * objSize, nthreads), nthreads, 0, defaultStream,
                       partIds, numObjects * objSize, 12311);
    SAFE_KERNEL_LAUNCH(setIds,
                       getNblocks(numObjects, nthreads), nthreads, 0, defaultStream,
                       objIds, numObjects, 239217);

    // Make a copy before deletion.
    ObjectData oldData{ov->local(), defaultStream};

    // Mark which to delete and delete.
    PinnedBuffer<int> maskBuffer{numObjects};
    for (int i = 0; i < numObjects; ++i)
        maskBuffer[i] = mask[i];
    maskBuffer.uploadToDevice(defaultStream);

    ObjectDeleter deleter;
    deleter.update(ov->local(), defaultStream);
    SAFE_KERNEL_LAUNCH(markParticles,
                       getNblocks(numObjects, nthreads), nthreads, 0, defaultStream,
                       deleter.handler(), maskBuffer.devPtr(), numObjects);

    deleter.deleteObjects(ov->local(), defaultStream);

    // Download new data and compare.
    ObjectData newData{ov->local(), defaultStream};
    CUDA_Check( cudaStreamSynchronize(defaultStream) );

    int sum = 0;
    for (int i = 0; i < numObjects; ++i) {
        if (mask[i])
            continue;
        compareObjects(oldData, newData, i, sum++);
    }
}

void test_large_delete(int N, double p, int objSize)
{
    // Was not sure if cub::ExclusiveSum works fine with `bool` markers that
    // get accumulated into `int`s. This shows it works fine.
    std::vector<int> mask(N);
    std::mt19937 gen;
    std::uniform_real_distribution<double> distr(0, 1);
    for (int i = 0; i < N; ++i)
        mask[i] = distr(gen) < p ? 1 : 0;
    test_delete(mask, objSize);
}


TEST (OBJECT_DELETER, DeleteObjects)
{
    test_delete({0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 123);  // Leave all.
    test_delete({1, 0, 0, 0, 0, 0, 0, 0, 0, 1}, 123);
    test_delete({0, 0, 1, 1, 0, 0, 1, 0, 0, 1}, 123);
    test_delete({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 123);  // Remove all.

    test_large_delete(1000, 0.7, 123);
    test_large_delete(5000, 0.3, 123);
    test_large_delete(10000, 0.9, 50);
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "object_deleter.log", 9);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
