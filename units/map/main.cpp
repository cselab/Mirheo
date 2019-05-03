#include <core/logger.h>
#include <core/pvs/packers/map.h>

#include <cstdio>
#include <gtest/gtest.h>

Logger logger;

TEST (MAP, Entry_backAndForth)
{
    const int ntries = 10000;
    
    for (int i = 0; i < ntries; ++i)
    {
        int bufId = rand() % 27;
        int id = rand() % (1<<(32-5));
        MapEntry m;
        
        m.setId(id);
        m.setBufId(bufId);
        
        ASSERT_EQ(id,    m.getId());
        ASSERT_EQ(bufId, m.getBufId());
    }
}

TEST (MAP, ThreadDispatch)
{
    const int ntries = 1000;
    
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
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
