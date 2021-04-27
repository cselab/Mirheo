// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "file_wrapper.h"
#include <mirheo/core/logger.h>
#include <mirheo/core/utils/strprintf.h>

namespace mirheo
{

FileWrapper::FileWrapper() = default;

FileWrapper::FileWrapper(const std::string& fname, const std::string& mode)
{
    if (open(fname, mode) != Status::Success)
    {
        die("Could not open the file \"%s\" in mode \"%s\".",
            fname.c_str(), mode.c_str());
    }
}

FileWrapper::FileWrapper(SpecialStream stream, bool forceFlushOnClose) {
    open(stream, forceFlushOnClose);
}

FileWrapper::~FileWrapper()
{
    close();
}

FileWrapper::FileWrapper(FileWrapper&& f)
{
    std::swap(file_, f.file_);
    std::swap(forceFlushOnClose_, f.forceFlushOnClose_);
}

FileWrapper& FileWrapper::operator=(FileWrapper&& f)
{
    std::swap(file_, f.file_);
    std::swap(forceFlushOnClose_, f.forceFlushOnClose_);
    return *this;
}

FileWrapper::Status FileWrapper::open(const std::string& fname, const std::string& mode)
{
    close();
    file_ = fopen(fname.c_str(), mode.c_str());

    if (file_ == nullptr)
        return Status::Failed;

    return Status::Success;
}

FileWrapper::Status FileWrapper::open(FileWrapper::SpecialStream stream, bool forceFlushOnClose)
{
    close();
    switch(stream)
    {
    case SpecialStream::Cout: file_ = stdout; break;
    case SpecialStream::Cerr: file_ = stderr; break;
    }

    forceFlushOnClose_ = forceFlushOnClose;

    return Status::Success;
}

void FileWrapper::close()
{
    if (forceFlushOnClose_ && (file_ == stdout || file_ == stderr))
        fflush(file_);

    if (file_ != stdout && file_ != stdout && file_ != nullptr)
        fclose(file_);

    file_ = nullptr;
    forceFlushOnClose_ = false;
}

void FileWrapper::fread(void *ptr, size_t size, size_t count) {
    size_t read = ::fread(ptr, size, count, file_);
    if (read != count)
    {
        die("Successfully read only %zu out of %zu element(s) of size %zuB.",
            read, count, size);
    }
}

} // namespace mirheo
