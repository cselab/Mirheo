#include <core/containers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/pvs/data_packers/data_packers.h>

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

Logger logger;

TEST (PACKERS_SIMPLE, particles)
{    
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
