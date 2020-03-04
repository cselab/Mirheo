#include "file_wrapper.h"
#include <mirheo/core/logger.h>

namespace mirheo
{

FileWrapper::FileWrapper(bool forceFlushOnClose) :
    forceFlushOnClose_(forceFlushOnClose)
{}

FileWrapper::FileWrapper(const std::string& fname, const std::string& mode,
                         bool forceFlushOnClose) :
    forceFlushOnClose_(forceFlushOnClose)
{
    if (open(fname, mode) != Status::Success) {
        die("Could not open the file \"%s\" in mode \"%s\".",
            fname.c_str(), mode.c_str());
    }
    needClose_ = true;
}

FileWrapper::~FileWrapper()
{
    close();
}

FileWrapper::FileWrapper(FileWrapper&& f) :
    FileWrapper(f.forceFlushOnClose_)
{
    std::swap(file_, f.file_);
    std::swap(needClose_, f.needClose_);
    std::swap(forceFlushOnClose_, f.forceFlushOnClose_);
}

FileWrapper& FileWrapper::operator=(FileWrapper&& f)
{
    std::swap(file_, f.file_);
    std::swap(needClose_, f.needClose_);
    std::swap(forceFlushOnClose_, f.forceFlushOnClose_);
    return *this;
}

FileWrapper::Status FileWrapper::open(const std::string& fname, const std::string& mode)
{
    if (needClose_) close();

    file_ = fopen(fname.c_str(), mode.c_str());

    if (file_ == nullptr)
        return Status::Failed;

    needClose_ = true;
    return Status::Success;
}

FileWrapper::Status FileWrapper::open(FileWrapper::SpecialStream stream)
{
    if (needClose_) close();

    switch(stream)
    {
    case SpecialStream::Cout: file_ = stdout; break;
    case SpecialStream::Cerr: file_ = stderr; break;
    }

    needClose_ = false;
    return Status::Success;
}

void FileWrapper::close()
{
    if (needClose_)
    {
        if (forceFlushOnClose_)
            fflush(file_);
        fclose(file_);
        needClose_ = false;
    }
}

} // namespace mirheo
