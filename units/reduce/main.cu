#include <mirheo/core/containers.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <numeric>
#include <random>

using namespace mirheo;

namespace ReduceKernels
{
__global__ void reduce(int n, const float *data, float *result)
{
    const auto i = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.f;

    if (i < n) val = data[i];

    val = warpReduce(val, [](float a, float b) {return a + b;});

    if (threadIdx.x % warpSize == 0)
        atomicAdd(result, val);
}
} // namespace ReduceKernels

static float reduceGPU(const PinnedBuffer<float>& data, cudaStream_t stream)
{
    PinnedBuffer<float> result(1);
    result.clear(stream);

    constexpr int nthreads = 128;
    const int nblocks = getNblocks(data.size(), nthreads);

    SAFE_KERNEL_LAUNCH(
        ReduceKernels::reduce,
        nblocks, nthreads, 0, stream,
        data.size(), data.devPtr(), result.devPtr());

    result.downloadFromDevice(stream);
    return result[0];
}

static float reduceCPU(const PinnedBuffer<float>& data)
{
    return std::accumulate(data.begin(), data.end(), 0.f);
}

PinnedBuffer<float> initData(int n, cudaStream_t stream, long seed=42)
{
    PinnedBuffer<float> data(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (auto& v : data)
        v = dis(gen);

    data.uploadToDevice(stream);
    return data;
}

static void testReduceRandom(int n, double tolerance=1e-6)
{
    const auto data = initData(n, defaultStream);

    const auto resGPU = reduceGPU(data, defaultStream);
    const auto resCPU = reduceCPU(data);

    const double err = math::abs(resCPU - resGPU) / math::abs(resCPU);

    ASSERT_LT(err, tolerance) << "failed: " << resGPU << " != " << resCPU;
}

TEST (REDUCE, oneWarp)
{
    testReduceRandom(32);
}

TEST (REDUCE, medium)
{
    testReduceRandom(1000);
}

TEST (REDUCE, large)
{
    testReduceRandom(1<<20, 5e-6);
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
