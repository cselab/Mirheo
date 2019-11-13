#pragma once

#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/macros.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>

#ifndef COMPILE_DEBUG_LVL
/**
 * Caps the maximum debug level of Logger at compile-time.
 * Typically overhead of NOT executed (due to low priority) logger calls
 * is small, and there is no need to remove debug output at compile time.
 */
#define COMPILE_DEBUG_LVL 10
#endif

namespace mirheo
{

/**
 * Class providing logging functionality with MPI support.
 *
 * Each MPI process writes to its own file, prefixing messages with time stamps
 * so that later the information may be combined and sorted.
 * Filenames have the following pattern, NNNNN is the MPI rank with leading zeros:
 * \c \<common_name\>_NNNNN.log
 *
 * Debug level governs which messages to log will be printed, the higher the level
 * the more stuff will be dumped.
 *
 * Every logging call has an importance level associated with it, which is compared
 * against the governing debug level, e.g. debug() importance is 4 and error()
 * importance is 1
 *
 * \code
 * Logger logger;
 * \endcode
 * has to be defined in one the objective file (typically main). Prior to any logging
 * a method init() must be called
 */
class Logger
{
public:

    /**
     * Set logger to write to files
     *
     * @param comm  relevant MPI communicator, typically \e MPI_COMM_WORLD
     * @param filename log files will be prefixed with \e filename: e.g. \e fname_<rank_with_leading_zeros>.log
     * @param debugLvl debug level
     */
    void init(MPI_Comm comm, const std::string& filename, int debugLvl = 3);

    /**
     * Set logger to write to a certain, already opened file.
     *
     * @param comm relevant MPI communicator, typically \e MPI_COMM_WORLD
     * @param fout file handler, must be opened, typically \e stdout or \e stderr
     * @param debugLvl debug level
     */
    void init(MPI_Comm comm, FileWrapper&& fout, int debugLvl = 3);

    int getDebugLvl() const noexcept {
        return runtimeDebugLvl;
    }
    void setDebugLvl(int debugLvl);

    /**
     * Main logging function.
     *
     * Construct a logger entry with time prefix, importance string,
     * filename and line number, and the message itself.
     *
     * And finally print the output to the file.
     *
     * This function is not supposed to be called directly, use appropriate
     * macros instead, e.g. #say(), #error(), #debug(), etc.
     *
     * \rst
     * .. attention::
     *    When the debug level is higher or equal to the _flushThreshold_ member
     *    variable (default 8), every message is flushed to disk immediately. This may
     *    increase the runtime SIGNIFICANTLY and only recommended to debug crashes
     * \endrst
     *
     * @param filename name of the current source file
     * @param line     line number of the source file
     * @param pattern  message pattern to be passed to \e printf
     * @param args     other relevant arguments to \e printf
     */
    void log(const char *key, const char *filename, int line, const char *pattern, ...) const;
    void logImpl(const char *key, const char *filename, int line, const char *pattern, va_list) const;

    void printStacktrace() const;
    
    /**
     * Special wrapper around log() that kills the application on
     * a fatal error
     * Print stack trace, error message, close the file and abort
     *
     * @param args forwarded to log()
     */
    void _die [[noreturn]](const char *filename, int line, const char *fmt, ...) const;

    /**
     * Wrapper around _die() that prints the current MPI error.
     *
     * @param filename name of the current source file
     * @param line  line number of the source file
     * @param code  error code (returned by MPI call)
     */
    void _MPI_die [[noreturn]](const char *filename, int line, int code) const;

    /**
     * Check the return code of an MPI function. Default error checking of MPI
     * should be turned off
     *
     * @param filename name of the current source file
     * @param line  line number of the source file
     * @param code  error code (returned by MPI call)
     */
    inline void _MPI_Check(const char *filename, const int line, const int code) const
    {
        if (code != MPI_SUCCESS)
            _MPI_die(filename, line, code);
    }

    /**
     * @param filename name of the current source file
     * @param line  line number of the source file
     * @param code  error code (returned by CUDA call)
     */
    inline void _CUDA_Check(const char *filename, const int line, cudaError_t code) const
    {
        if (code != cudaSuccess)
            _die(filename, line, "%s", cudaGetErrorString(code));
    }

private:
    int runtimeDebugLvl {0};  ///< debug level defined at runtime through setDebugLvl

    static constexpr int flushThreshold = 8; ///< value of debug level starting with which every
                                             ///< message will be flushed to disk immediately

    static constexpr int numLogsBetweenFlushes = 32;
    mutable int numLogsSinceLastFlush {0};

    mutable FileWrapper fout {true};
    int rank {-1};
};

/// Log with a runtime check of importance
#define MIRHEO_LOG_IMPL(LEVEL, KEY, ...) \
    do { \
        if (::mirheo::logger.getDebugLvl() >= (LEVEL)) \
            ::mirheo::logger.log((KEY), __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (0)

/// Unconditionally print to log, debug level is not checked here
#define   say(...)  MIRHEO_LOG_IMPL(1, "INFO", ##__VA_ARGS__)

#if COMPILE_DEBUG_LVL >= 0
/// Report a fatal error and abort
#define   MIRHEO_DIE(...)  ::mirheo::logger._die(__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define   MIRHEO_DIE(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 1
/// Report a serious error
#define error(...)  MIRHEO_LOG_IMPL(1, "ERROR", ##__VA_ARGS__)
#else
#define error(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 2
/// Report a warning
#define  warn(...)  MIRHEO_LOG_IMPL(2, "WARNING", ##__VA_ARGS__)
#else
#define  warn(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 3
/// Report certain valuable information
#define  info(...)  MIRHEO_LOG_IMPL(3, "INFO", ##__VA_ARGS__)
#else
#define  info(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 4
/// Print debug output
#define debug(...)  MIRHEO_LOG_IMPL(4, "DEBUG", ##__VA_ARGS__)
#else
#define debug(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 5
/// Print more debug
#define debug2(...)  MIRHEO_LOG_IMPL(5, "DEBUG", ##__VA_ARGS__)
#else
#define debug2(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 6
/// Print yet more debug
#define debug3(...)  MIRHEO_LOG_IMPL(6, "DEBUG", ##__VA_ARGS__)
#else
#define debug3(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 7
/// Print ultimately verbose debug. God help you scrambling through all the output
#define debug4(...)  MIRHEO_LOG_IMPL(7, "DEBUG", ##__VA_ARGS__)
#else
#define debug4(...)  do { } while(0)
#endif

/// Check an MPI call, call #die() if it fails
#define MIRHEO_MPI_CHECK(command)  ::mirheo::logger._MPI_Check (__FILE__, __LINE__, command)

/// Check a CUDA call, call #die() if it fails
#define MIRHEO_CUDA_CHECK(command) ::mirheo::logger._CUDA_Check(__FILE__, __LINE__, command)

/// Shorthands for the macros above.
#define die         MIRHEO_DIE
#define MPI_Check   MIRHEO_MPI_CHECK
#define CUDA_Check  MIRHEO_CUDA_CHECK

/**
 * A common `Logger` object for all Mirheo and all potential extension files.
 * The instance is defined in `logger.cpp`.
 */
extern Logger logger;

} // namespace mirheo
