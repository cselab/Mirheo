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
    
    explicit FileWrapper(bool forceFlushOnClose = false);
    ~FileWrapper();

    FileWrapper           (const FileWrapper&) = delete;
    FileWrapper& operator=(const FileWrapper&) = delete;
    
    FileWrapper           (FileWrapper&&);
    FileWrapper& operator=(FileWrapper&&);

    Status open(const std::string& fname, const std::string& mode);
    Status open(SpecialStream stream);
    
    FILE* get() {return file_;}

    void close();
    
private:
    
    FILE *file_ {nullptr};
    bool needClose_ {false};
    bool forceFlushOnClose_;
};

} // namespace mirheo
