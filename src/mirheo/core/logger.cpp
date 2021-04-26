// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "logger.h"

#include <mirheo/core/utils/path.h>
#include <mirheo/core/utils/stacktrace_explicit.h>
#include <mirheo/core/utils/strprintf.h>

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <ctime>
#include <sstream>
#include <stdexcept>

namespace mirheo
{

Logger logger;

int getDefaultDebugLvl() {
    const char *var = std::getenv("MIRHEO_DEBUG_LEVEL");
    if (var != nullptr && var[0] != '\0') {
        int lvl;
        if (1 == sscanf(var, "%d", &lvl))
            return lvl;
        fprintf(stderr, "MIRHEO_DEBUG_LEVEL should be an integer, got \"%s\". Ignoring.", var);
    }
    return 3;  // The default value if no environment variable is set.
}

static void printStacktrace(FILE *fout)
{
    std::ostringstream strace;
    stacktrace::getStacktrace(strace);
    fwrite(strace.str().c_str(), sizeof(char), strace.str().size(), fout);
}

Logger::Logger() = default;
Logger::~Logger() = default;

void Logger::init(MPI_Comm comm, const std::string& fname, int debugLvl)
{
    MPI_Comm_rank(comm, &rank_);
    constexpr int zeroPadding = 5;
    const std::string rankStr = createStrZeroPadded(rank_, zeroPadding);

    const auto pos   = fname.find_last_of('.');
    const auto start = fname.substr(0, pos);
    const auto end   = fname.substr(pos);

    const auto status = fout_.open(start + "_" + rankStr + end, "w");

    if (status != FileWrapper::Status::Success)
    {
        fprintf(stderr, "Logger file '%s' could not be open.\n", fname.c_str());
        exit(1);
    }

    setDebugLvl(debugLvl);

    stacktrace::registerSignals();
}

void Logger::init(MPI_Comm comm, FileWrapper&& fout, int debugLvl)
{
    MPI_Comm_rank(comm, &rank_);
    this->fout_ = std::move(fout);

    setDebugLvl(debugLvl);
}

void Logger::setDebugLvl(int debugLvl)
{
    if (debugLvl < 0)
        debugLvl = getDefaultDebugLvl();
    runtimeDebugLvl_ = std::min(debugLvl, COMPILE_DEBUG_LVL);
    if (runtimeDebugLvl_ >= 1) {
        log("INFO", __FILE__, __LINE__,
            "Compiled with maximum debug level %d", COMPILE_DEBUG_LVL);
        log("INFO", __FILE__, __LINE__,
            "Debug level requested %d, set to %d", debugLvl, runtimeDebugLvl_);
    }
}

void Logger::log(const char *key, const char *filename, int line, const char *fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    _logImpl(key, filename, line, fmt, args);
    va_end(args);
}

void Logger::_logImpl(const char *key, const char *filename, int line, const char *fmt, va_list args) const {
    if (!fout_.get())
    {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        printStacktrace(stderr);
        fprintf(stderr, "\n\nLogger file is not set but tried to be used at %s:%d from rank %d"
                " with the following message:\n", filename, line, world_rank);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        throw std::runtime_error("Logger used before initialization. Message was printed to stderr.");
    }

    using namespace std::chrono;
    auto now   = system_clock::now();
    auto now_c = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    char time[16];    // "%T" --> "HH:MM:SS".
    size_t len = std::strftime(time, sizeof(time), "%T", std::localtime(&now_c));
    (void)len;
    assert(len > 0);  // Returns 0 if the format does not fit, which is impossible here.

    // It's not really possible to extend a va_list, so we have to add an
    // fprintf before and after to print extra formatting. It may be necessary
    // to replace this with first constructing a string with (v)s(n)printf and
    // then fprintf-ing it once.
    fprintf(fout_.get(), "%s:%03d  Rank %04d %7s at %s:%d ",
            time, (int)ms.count(), rank_, key, filename, line);
    vfprintf(fout_.get(), fmt, args);
    fprintf(fout_.get(), "\n");

    ++numLogsSinceLastFlush_;

    bool needToFlush = runtimeDebugLvl_  >= flushThreshold_ &&
                       COMPILE_DEBUG_LVL >= flushThreshold_;
    needToFlush = needToFlush || (numLogsSinceLastFlush_ > numLogsBetweenFlushes_);

    if (needToFlush)
    {
        fflush(fout_.get());
        numLogsSinceLastFlush_ = 0;
    }
}

void Logger::_die [[noreturn]](const char *filename, int line, const char *fmt, ...) const
{
    va_list args;
    va_start(args, fmt);
    _logImpl("", filename, line, fmt, args);
    va_end(args);

    printStacktrace(fout_.get());
    fout_.close();

    // http://stackoverflow.com/a/26221725  (modified)
    va_start(args, fmt);
    std::string error = vstrprintf(fmt, args);
    va_end(args);

    throw std::runtime_error("Mirheo has encountered a fatal error and will quit now.\n"
                             "The error message follows, and more details can be found in the log\n"
                             "***************************************\n"
                             "\t" + error + "\n"
                             "***************************************");
}

void Logger::_CUDA_die [[noreturn]](const char *filename, int line, cudaError_t code) const
{
    _die(filename, line, "CUDA Error on rank %d: %s", rank_, cudaGetErrorString(code));
}

void Logger::_MPI_die [[noreturn]](const char *filename, int line, int code) const
{
    char buf[MPI_MAX_ERROR_STRING];
    int nchar;
    MPI_Error_string(code, buf, &nchar);

    _die(filename, line, "MPI Error on rank %d: %s", rank_, buf);
}

} // namespace mirheo
