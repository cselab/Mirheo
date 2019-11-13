#include <mirheo/core/containers.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace mirheo;

// assume only one block of 32 threads
__global__ void inclusiveScan(int *data)
{
    int i = threadIdx.x;
    data[i] = warpInclusiveScan(data[i]);
}

__global__ void exclusiveScan(int *data)
{
    int i = threadIdx.x;
    data[i] = warpExclusiveScan(data[i]);
}

static void inclusiveScan(std::vector<int>& data)
{
    for (size_t i = 1; i < data.size(); ++i)
        data[i] += data[i-1];
}

static void exclusiveScan(std::vector<int>& data)
{
    int s = 0, tmp;
    for (size_t i = 0; i < data.size(); ++i) {
        tmp = data[i];
        data[i] = s;
        s += tmp;
    }
}

TEST (WARP_SCAN, inclusive)
{
    cudaStream_t defaultStream = 0;

    constexpr int N = 32; // warp size
    std::vector<int> hData(N);
    PinnedBuffer<int> dData(N);

    for (int i = 0; i < N; ++i)
        hData[i] = dData[i] = i+3;

    dData.uploadToDevice(defaultStream);

    SAFE_KERNEL_LAUNCH(
        inclusiveScan,
        1, N, 0, defaultStream,
        dData.devPtr() );

    dData.downloadFromDevice(defaultStream);

    inclusiveScan(hData);

    for (int i = 0; i < N; ++i)
        ASSERT_EQ(hData[i], dData[i]);
}

TEST (WARP_SCAN, exclusive)
{
    cudaStream_t defaultStream = 0;

    constexpr int N = 32; // warp size
    std::vector<int> hData(N);
    PinnedBuffer<int> dData(N);

    for (int i = 0; i < N; ++i)
        hData[i] = dData[i] = i+3;

    dData.uploadToDevice(defaultStream);

    SAFE_KERNEL_LAUNCH(
        exclusiveScan,
        1, N, 0, defaultStream,
        dData.devPtr() );

    dData.downloadFromDevice(defaultStream);

    exclusiveScan(hData);

    for (int i = 0; i < N; ++i)
        ASSERT_EQ(hData[i], dData[i]);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
