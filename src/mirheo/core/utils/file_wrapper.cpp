#include "file_wrapper.h"

namespace mirheo
{

FileWrapper::FileWrapper(bool forceFlushOnClose) :
    forceFlushOnClose_(forceFlushOnClose)
{}

FileWrapper::~FileWrapper()
{
    close();
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
