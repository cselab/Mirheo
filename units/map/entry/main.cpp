#include <core/logger.h>
#include <core/pvs/packers/map.h>

#include <cstdio>
#include <gtest/gtest.h>

Logger logger;

TEST (MAP_ENTRY, backAndForth)
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

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
