#pragma once

#include <cstdio>
#include <string>

namespace mirheo
{

class FileWrapper
{
public:
    
    enum class SpecialStream {Cout, Cerr};
    enum class Status {Success, Failed};

    FileWrapper();
    FileWrapper(const std::string& fname, const std::string& mode);
    ~FileWrapper();

    FileWrapper           (const FileWrapper&) = delete;
    FileWrapper& operator=(const FileWrapper&) = delete;
    
    FileWrapper           (FileWrapper&&);
    FileWrapper& operator=(FileWrapper&&);

    Status open(const std::string& fname, const std::string& mode);
    Status open(SpecialStream stream, bool forceFlushOnClose);
    
    FILE* get() {return file_;}

    void close();
    
private:
    FILE *file_ {nullptr};
    bool forceFlushOnClose_{false};
};

} // namespace mirheo
