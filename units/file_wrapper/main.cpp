#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/logger.h>

#include <gtest/gtest.h>

using namespace mirheo;

TEST (FILE_WRAPPER, stdout )
{
    testing::internal::CaptureStdout();
    FileWrapper stream;
    stream.open(FileWrapper::SpecialStream::Cout);
    fprintf(stream.get(), "Hello");
    const std::string output = testing::internal::GetCapturedStdout();
    ASSERT_EQ(output, "Hello"); 
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
