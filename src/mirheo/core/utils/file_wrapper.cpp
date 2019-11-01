#include "file_wrapper.h"

namespace mirheo
{

FileWrapper::FileWrapper(bool forceFlushOnClose) :
    forceFlushOnClose(forceFlushOnClose)
{}

FileWrapper::~FileWrapper()
{
    close();
}

FileWrapper::Status FileWrapper::open(const std::string& fname, const std::string& mode)
{
    if (needClose) close();

    file = fopen(fname.c_str(), mode.c_str());

    if (file == nullptr)
        return Status::Failed;

    needClose = true;
    return Status::Success;
}

FileWrapper::Status FileWrapper::open(FileWrapper::SpecialStream stream)
{
    if (needClose) close();

    switch(stream)
    {
    case SpecialStream::Cout: file = stdout; break;
    case SpecialStream::Cerr: file = stderr; break;
    }

    needClose = false;
    return Status::Success;
}

void FileWrapper::close()
{
    if (needClose)
    {
        if (forceFlushOnClose) fflush(file);
        fclose(file);
        needClose = false;
    }
}

} // namespace mirheo
