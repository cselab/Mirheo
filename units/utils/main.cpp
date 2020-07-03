#include <mirheo/core/logger.h>
#include <mirheo/core/utils/path.h>

#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace mirheo;

TEST (UTILS, createStrZeroPadded)
{
    ASSERT_EQ(mirheo::createStrZeroPadded(42, 5), "00042");
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

TEST (UTILS, makePath)
{
    ASSERT_EQ(mirheo::makePath("path"), "path/");
    ASSERT_EQ(mirheo::makePath("path/"), "path/");
    ASSERT_EQ(mirheo::makePath("this/is/more/complex/path"), "this/is/more/complex/path/");
    ASSERT_EQ(mirheo::makePath("this/is/more/complex/path/"), "this/is/more/complex/path/");
}

TEST (UTILS, getParentPath)
{
    ASSERT_EQ(mirheo::getParentPath("path/file.h5"), "path/");
    ASSERT_EQ(mirheo::getParentPath("this/is/more/complex/path/file.h5"), "this/is/more/complex/path/");
    ASSERT_EQ(mirheo::getParentPath("file.h5"), "");
    ASSERT_EQ(mirheo::getParentPath("just/a/path/"), "just/a/path/");
}

TEST (UTILS, getBaseName)
{
    ASSERT_EQ(mirheo::getBaseName("path/file.h5"), "file.h5");
    ASSERT_EQ(mirheo::getBaseName("more/complex/path/file.h5"), "file.h5");
    ASSERT_EQ(mirheo::getBaseName("file.h5"), "file.h5");
}

TEST (UTILS, joinPaths)
{
    ASSERT_EQ(mirheo::joinPaths("path/", "file.h5"), "path/file.h5");
    ASSERT_EQ(mirheo::joinPaths("path", "file.h5"), "path/file.h5");
    ASSERT_EQ(mirheo::joinPaths("", "file.h5"), "file.h5");
    ASSERT_EQ(mirheo::joinPaths("path/", ""), "path/");
    ASSERT_EQ(mirheo::joinPaths("path", ""), "path/");
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
