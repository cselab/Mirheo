#include "logger.h"

#include <mirheo/core/utils/stacktrace_explicit.h>
#include <mirheo/core/utils/folders.h>

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <ctime>
#include <sstream>
#include <stdexcept>

namespace mirheo
{

Logger logger;


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

void Logger::setDebugLvl(int debugLvl)
{
    runtimeDebugLvl = std::max(std::min(debugLvl, COMPILE_DEBUG_LVL), 0);
    if (runtimeDebugLvl >= 1) {
        log("INFO", __FILE__, __LINE__,
            "Compiled with maximum debug level %d", COMPILE_DEBUG_LVL);
        log("INFO", __FILE__, __LINE__,
            "Debug level requested %d, set to %d", debugLvl, runtimeDebugLvl);
    }
}

void Logger::log(const char *key, const char *filename, int line, const char *fmt, ...) const {
    va_list args;
    va_start(args, fmt);
    logImpl(key, filename, line, fmt, args);
    va_end(args);
}

void Logger::logImpl(const char *key, const char *filename, int line, const char *fmt, va_list args) const {
    if (!fout.get())
    {
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        fprintf(stderr, "Logger file is not set but tried to be used at %s:%d from rank %d"
                " with the following message:\n", filename, line, world_rank);
        vfprintf(stderr, fmt, args);
        fprintf(stderr, "\n");
        exit(1);
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
    fprintf(fout.get(), "%s:%03d  Rank %04d %7s at %s:%d ",
            time, (int)ms.count(), rank, key, filename, line);
    vfprintf(fout.get(), fmt, args);
    fprintf(fout.get(), "\n");

    bool needToFlush = runtimeDebugLvl   >= flushThreshold &&
                       COMPILE_DEBUG_LVL >= flushThreshold;
    needToFlush = needToFlush || (numLogsSinceLastFlush > numLogsBetweenFlushes);

    if (needToFlush)
    {
        fflush(fout.get());
        numLogsSinceLastFlush = 0;
    }
}

/// std::string variant of vsprintf.
static std::string vstrprintf(const char *fmt, va_list args) {
    va_list args2;
    va_copy(args2, args);

    const int size = vsnprintf(nullptr, 0, fmt, args) + 1;

    std::string result(size, '_');
    vsnprintf(&result[0], size + 1, fmt, args2);
    return result;
}

void Logger::_die [[noreturn]](const char *filename, int line, const char *fmt, ...) const
{
    va_list args;
    va_start(args, fmt);
    logImpl("", filename, line, fmt, args);
    va_end(args);

    printStacktrace();
    fout.close();

    // http://stackoverflow.com/a/26221725  (modified)
    va_start(args, fmt);
    std::string error = vstrprintf(fmt, args);
    va_end(args);
    error.pop_back(); // remove '\0'

    throw std::runtime_error("Mirheo has encountered a fatal error and will quit now.\n"
                             "The error message follows, and more details can be found in the log\n"
                             "***************************************\n"
                             "\t" + error + "\n"
                             "***************************************");
}

void Logger::_CUDA_die [[noreturn]](const char *filename, int line, cudaError_t code) const
{
    _die(filename, line, "CUDA Error on rank %d: %s", rank, cudaGetErrorString(code));
}

void Logger::_MPI_die [[noreturn]](const char *filename, int line, int code) const
{
    char buf[MPI_MAX_ERROR_STRING];
    int nchar;
    MPI_Error_string(code, buf, &nchar);

    _die(filename, line, "MPI Error on rank %d: %s", rank, buf);
}

void Logger::printStacktrace() const
{
    std::ostringstream strace;
    pretty_stacktrace(strace);
    fwrite(strace.str().c_str(), sizeof(char), strace.str().size(), fout.get());
}

} // namespace mirheo
