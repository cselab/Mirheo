#include <mirheo/core/logger.h>
#include <mirheo/core/datatypes.h>

#include <cstdio>
#include <gtest/gtest.h>

Logger logger;

TEST (ID64, backAndForth)
{
    int64_t id = (42l << 32) + 13l;
    Particle p;

    p.setId(id);
    
    ASSERT_EQ(id, p.getId());
}

TEST (ID64, subids)
{
    int64_t low  = 12345l;
    int64_t high = 54321l;
    int64_t id = (high << 32) + low;

    Particle p;
    p.setId(id);
    
    ASSERT_EQ(p.i1, low);
    ASSERT_EQ(p.i2, high);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
