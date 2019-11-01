#include <mirheo/core/containers.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <gtest/gtest.h>
#include <extern/variant/include/mpark/variant.hpp>
#include <variant/variant.h>

#include <vector>

using VarType    = mpark::variant<int, float>;
using VarTypePtr = variant::variant<int*, float*>;

using namespace mirheo;

namespace mirheo { Logger logger; }

__HD__ inline int eval(int a)
{
    return a * a;
}

__HD__ inline float eval(float a)
{
    return 1.f - a * 0.5f;
}

__global__ void variantOperation(int n, const VarTypePtr dataVar, void *result)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    variant::apply_visitor([&](auto dataPtr)
    {
        auto dst = reinterpret_cast<decltype(dataPtr)>(result);
        dst[i] = eval(dataPtr[i]);
    }, dataVar);        
}

TEST (Variant, cpu)
{
    int   i = 42;
    float f = 3.14;
    
    VarType var;

    auto check = [](auto ref, VarType var) -> bool
    {
        return mpark::visit([&](auto val)
        {
            return ref == eval(val);
        }, var);
    };

    var = i;
    ASSERT_TRUE(check(eval(i), var));

    var = f;
    ASSERT_TRUE(check(eval(f), var));
}

TEST (Variant, gpu)
{
    int n = 128;
    PinnedBuffer<int> intData(n), intRes(n);
    PinnedBuffer<float> floatData(n), floatRes(n);
    std::vector<int> intRef(n);
    std::vector<float> floatRef(n);

    for (auto& d : intData)
        d = rand();

    for (auto& d: floatData)
        d = rand() / 5.f;

    intData  .uploadToDevice(defaultStream);
    floatData.uploadToDevice(defaultStream);

    for (int i = 0; i < n; ++i)
    {
        intRef  [i] = eval(  intData[i]);
        floatRef[i] = eval(floatData[i]);
    }

    VarTypePtr dataVar;

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    
    dataVar = intData.devPtr();
    SAFE_KERNEL_LAUNCH(variantOperation,
                       nblocks, nthreads, 0, defaultStream,
                       n, dataVar, intRes.devPtr());

    dataVar = floatData.devPtr();
    SAFE_KERNEL_LAUNCH(variantOperation,
                       nblocks, nthreads, 0, defaultStream,
                       n, dataVar, floatRes.devPtr());


    intRes  .downloadFromDevice(defaultStream);
    floatRes.downloadFromDevice(defaultStream);
    
    for (int i = 0; i < n; ++i)
    {
        ASSERT_EQ(intRes[i], intRef[i]);
        ASSERT_EQ(floatRes[i], floatRef[i]);
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
