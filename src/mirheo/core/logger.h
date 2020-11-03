// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/macros.h>

#include <cuda_runtime.h>
#include <mpi.h>
#include <string>

#ifndef COMPILE_DEBUG_LVL
/** Caps the maximum debug level of Logger at compile-time.
    Typically overhead of NOT executed (due to low priority) logger calls
    is small, and there is no need to remove debug output at compile time.
 */
#define COMPILE_DEBUG_LVL 10
#endif

namespace mirheo
{

/** \brief logging functionality with MPI support.

    Each MPI process writes to its own file, prefixing messages with time stamps
    so that later the information may be combined and sorted.
    Filenames have the following pattern, NNNNN is the MPI rank with leading zeros:
    \c \<common_name\>_NNNNN.log

    Debug level governs which messages to log will be printed (a higher level will
    dump more log messages).

    Every logging call has an importance level associated with it, which is compared
    against the governing debug level, e.g. debug() importance is 4 and error()
    importance is 1.

    \code
    Logger logger;
    \endcode
    has to be defined in one the objective file (typically the one that contains main()).
    Prior to any logging the method init() must be called.
 */
class Logger
{
public:

    /** \brief Setup the logger object
        \param [in] comm MPI communicator that contains all ranks that will use the logger
        \param [in] filename log files will be prefixed with \e filename: e.g. \e filename_<rank_with_leading_zeros>.log
        \param [in] debugLvl debug level

        Must be called before any logging method.
     */
    void init(MPI_Comm comm, const std::string& filename, int debugLvl = 3);

    /** \brief Setup the logger object to write to a given file.
        \param [in] comm  MPI communicator that contains all ranks that will use the logger
        \param [in] fout file handler, must be open, typically \e stdout or \e stderr
        \param [in] debugLvl debug level
     */
    void init(MPI_Comm comm, FileWrapper&& fout, int debugLvl = 3);

    /// return The current debug level
    int getDebugLvl() const noexcept
    {
        return runtimeDebugLvl_;
    }

    /** \brief set the debug level
        \param [in] debugLvl debug level
    */
    void setDebugLvl(int debugLvl);

    /** \brief Main logging function.

        Construct and dump a log entry with time prefix, importance string,
        filename and line number, and the message itself.

        This function is not supposed to be called directly, use appropriate
        macros instead, e.g. say(), error(), debug().

        \rst
        .. warning::
            When the debug level is higher or equal to the \c `flushThreshold_` member
            variable (default 8), every message is flushed to disk immediately. This may
            increase the runtime significantly and only recommended to debug crashes.
        \endrst

        \param [in] key The importance string, e.g. LOG or WARN
        \param [in] filename name of the current source file
        \param [in] line     line number in the current source file
        \param [in] pattern  message pattern to be passed to \e printf
     */
    void log [[gnu::format(printf, 5, 6)]] (
            const char *key, const char *filename, int line, const char *pattern, ...) const;

    /** \brief Calls log() and kills the application on a fatal error

        Print stack trace, error message, close the file and abort.
        See log() for parameters.
     */
    void _die [[gnu::format(printf, 4, 5)]] [[noreturn]] (
            const char *filename, int line, const char *fmt, ...) const;

    /** \brief Calls _die() with the error message corresponding to the given CUDA error code.
        \param [in] filename name of the current source file
        \param [in] line line number in the current source file
        \param [in] code CUDA error code (returned by a CUDA call)
     */
    void _CUDA_die [[noreturn]] (const char *filename, int line, cudaError_t code) const;

    /** \brief Calls _die() with the error message corresponding to the given MPI error code.
        \param [in] filename name of the current source file
        \param [in] line line number in the current source file
        \param [in] code MPI error code (returned by an MPI call)
     */
    void _MPI_die [[noreturn]] (const char *filename, int line, int code) const;

    /// check a CUDA error call and call _CUDA_die() in case of error
    inline void _CUDA_Check(const char *filename, const int line, cudaError_t code) const
    {
        if (code != cudaSuccess)
            _CUDA_die(filename, line, code);
    }

    /// check an MPI error call and call _MPI_die() in case of error
    inline void _MPI_Check(const char *filename, const int line, const int code) const
    {
        if (code != MPI_SUCCESS)
            _MPI_die(filename, line, code);
    }

private:
    void _logImpl(const char *key, const char *filename, int line, const char *pattern, va_list) const;

private:
    int runtimeDebugLvl_ {0};  ///< debug level defined at runtime through setDebugLvl

    static constexpr int flushThreshold_ = 8; ///< value of debug level starting with which every
                                             ///< message will be flushed to disk immediately

    static constexpr int numLogsBetweenFlushes_ = 32;
    mutable int numLogsSinceLastFlush_ {0};

    mutable FileWrapper fout_;
    int rank_ {-1};
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

/// Check an MPI call, call Logger::_die() if it fails
#define MIRHEO_MPI_CHECK(command)  ::mirheo::logger._MPI_Check (__FILE__, __LINE__, command)

/// Check a CUDA call, call Logger::_die() if it fails
#define MIRHEO_CUDA_CHECK(command) ::mirheo::logger._CUDA_Check(__FILE__, __LINE__, command)

/// Shorthands for the macros above.
#define die         MIRHEO_DIE
#define MPI_Check   MIRHEO_MPI_CHECK
#define CUDA_Check  MIRHEO_CUDA_CHECK

/**  The common `Logger` object for all Mirheo and all potential extension files.
     The instance is defined in `logger.cpp`.
 */
extern Logger logger;

} // namespace mirheo
