#include "logger.h"

#include <mirheo/core/utils/stacktrace_explicit.h>
#include <mirheo/core/utils/folders.h>

#include <chrono>
#include <iomanip>

namespace mirheo
{

void Logger::init(MPI_Comm comm, const std::string& fname, int debugLvl)
{
    MPI_Comm_rank(comm, &rank);
    constexpr int zeroPadding = 5;
    const std::string rankStr = getStrZeroPadded(rank, zeroPadding);

    const auto pos   = fname.find_last_of('.');
    const auto start = fname.substr(0, pos);
    const auto end   = fname.substr(pos);

    const auto status = fout.open(start + "_" + rankStr + end, "w");

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

std::string Logger::makeIntro(const char *fname, int lnum, const char *pattern) const
{
    using namespace std::chrono;
    auto now   = system_clock::now();
    auto now_c = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream tmout;
    tmout << std::put_time(std::localtime(&now_c), "%T") << ':'
          << std::setfill('0') << std::setw(3) << ms.count();

    const std::string intro = tmout.str() + "   " + std::string("Rank %04d %7s at ")
        + fname + ":" + std::to_string(lnum) + "  " + pattern + "\n";

    return intro;
}

void Logger::printStacktrace() const
{
    std::ostringstream strace;
    pretty_stacktrace(strace);
    fwrite(strace.str().c_str(), sizeof(char), strace.str().size(), fout.get());
}

} // namespace mirheo
