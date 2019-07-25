#pragma once

#include <cstdio>
#include <string>

class FileWrapper
{
public:
    
    enum class SpecialStream {Cout, Cerr};
    enum class Status {Success, Failed};
    
    FileWrapper() = default;
    ~FileWrapper();

    FileWrapper           (const FileWrapper&) = delete;
    FileWrapper& operator=(const FileWrapper&) = delete;
    
    FileWrapper           (FileWrapper&&) = default;
    FileWrapper& operator=(FileWrapper&&) = default;

    Status open(const std::string& fname, const std::string& mode);
    Status open(SpecialStream stream);
    
    FILE* get() {return file;}

    void close();
    
private:
    
    FILE *file {nullptr};
    bool needClose {false};
};
