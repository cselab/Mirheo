#include <mirheo/core/logger.h>
#include <mirheo/core/utils/folders.h>

#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace mirheo;

TEST (UTILS, getStrZeroPadded)
{
    ASSERT_EQ(mirheo::getStrZeroPadded(42, 5), "00042");
}

TEST (UTILS, splitByDelim)
{
    std::vector<std::string> list = {"this", "is", "a", "message"};
    const char sep = '_';

    std::string str = list.front();
    for (size_t i = 1; i < list.size(); ++i)
        str += sep + list[i];

    auto splitted = mirheo::splitByDelim(str, sep);
    
    ASSERT_EQ(splitted.size(), list.size());

    for (size_t i = 0; i < list.size(); ++i)
        ASSERT_EQ(splitted[i], list[i]);
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
