#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/logger.h>

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>

using namespace mirheo;

static const std::string content = "content test\nnothing to see here\n";

static std::string getFileContent(const std::string& fileName)
{
    std::ifstream f(fileName);
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

TEST (FILE_WRAPPER, stdout )
{
    testing::internal::CaptureStdout();
    FileWrapper stream;
    stream.open(FileWrapper::SpecialStream::Cout, true);
    fprintf(stream.get(), "%s", content.c_str());
    const std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output, content); 
}

TEST (FILE_WRAPPER, dump_file )
{
    const std::string fname = "test_dump.txt";

    FileWrapper stream;
    stream.open(fname, "w");
    fprintf(stream.get(), "%s", content.c_str());
    fflush(stream.get());

    const std::string dumped = getFileContent(fname);
    ASSERT_EQ(dumped, content); 
}

TEST (FILE_WRAPPER, move_dumper )
{
    const std::string fname = "test_dump.txt";

    FileWrapper stream0, stream1;
    stream0.open(fname, "w");
    stream1 = std::move(stream0);
    fprintf(stream1.get(), "%s", content.c_str());
    fflush(stream1.get());

    const std::string dumped = getFileContent(fname);
    ASSERT_EQ(dumped, content); 
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
