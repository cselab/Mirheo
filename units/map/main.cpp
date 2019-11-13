#include <mirheo/core/logger.h>
#include <mirheo/core/exchangers/utils/map.h>

#include <cstdio>
#include <gtest/gtest.h>

using namespace mirheo;

TEST (MAP, Entry_backAndForth)
{
    const int ntries = 10000;
    
    for (int i = 0; i < ntries; ++i)
    {
        const int bufId = rand() % 27;
        const int id = rand() % (1<<(32-5));
        const MapEntry m (id, bufId);
        
        ASSERT_EQ(id,    m.getId());
        ASSERT_EQ(bufId, m.getBufId());
    }
}

TEST (MAP, ThreadDispatch)
{
    const int ntries = 10000;
    
    for (int i = 0; i < ntries; ++i)
    {
        const int nBuffers = 27;
        const int maxSize = 42;
        int sizes[nBuffers], offsets[nBuffers+1] = {0};

        for (int i = 0; i < nBuffers; ++i) sizes[i] = rand() % maxSize;
        for (int i = 0; i < nBuffers; ++i) offsets[i+1] = offsets[i] + sizes[i];

        int tid = rand() % offsets[nBuffers];

        int buffId = dispatchThreadsPerBuffer(nBuffers, offsets, tid);
        
        ASSERT_LE(offsets[buffId], tid);
        ASSERT_GT(offsets[buffId+1], tid);
        ASSERT_LT(buffId, nBuffers);
        ASSERT_GE(buffId, 0);
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
