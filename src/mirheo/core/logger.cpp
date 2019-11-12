#include "logger.h"

namespace mirheo
{

void Logger::init(MPI_Comm comm, const std::string& fname, int debugLvl)
{
    MPI_Comm_rank(comm, &rank);
    constexpr int zeroPadding = 5;
    std::string rankStr = getStrZeroPadded(rank, zeroPadding);

    auto pos   = fname.find_last_of('.');
    auto start = fname.substr(0, pos);
    auto end   = fname.substr(pos);

    auto status = fout.open(start+"_"+rankStr+end, "w");

    if (status != FileWrapper::Status::Success)
    {
        fprintf(stderr, "Logger file '%s' could not be open.\n", fname.c_str());
        exit(1);
    }

    setDebugLvl(debugLvl);

    register_signals();
}

void Logger::init(MPI_Comm comm, FileWrapper&& fout, int debugLvl)
{
    MPI_Comm_rank(comm, &rank);
    this->fout = std::move(fout);
    
    setDebugLvl(debugLvl);
    lastFlushed = std::chrono::system_clock::now();
}

int Logger::getDebugLvl() const
{
    return runtimeDebugLvl;
}

void Logger::setDebugLvl(int debugLvl)
{
    runtimeDebugLvl = std::max(std::min(debugLvl, COMPILE_DEBUG_LVL), 0);
    log<1>("INFO", __FILE__, __LINE__,
           "Compiled with maximum debug level %d", COMPILE_DEBUG_LVL);
    log<1>("INFO", __FILE__, __LINE__,
           "Debug level requested %d, set to %d", debugLvl, runtimeDebugLvl);
}

} // namespace mirheo
