#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <cstdio>
#include <string>

#include <core/utils/stacktrace_explicit.h>

#include <mpi.h>
#include <cuda_runtime.h>

#ifndef COMPILE_DEBUG_LVL
/**
 * Caps the maximum debug level of Logger at compile-time.
 * Typically overhead of NOT executed (due to low priority) logger calls
 * is small, and there is no need to remove debug output at compile time.
 */
#define COMPILE_DEBUG_LVL 10
#endif


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

    /// Constructor doesn't do nothing
    Logger() {}

    /// Flush and close file
    ~Logger()
    {
        if (fout != nullptr)
        {
            fflush(fout);
            fclose(fout);
        }
    }


    /**
     * Set logger to write to files
     *
     * @param comm  relevant MPI communicator, typically \e MPI_COMM_WORLD
     * @param fname log files will be prefixed with \e fname: e.g. \e fname_<rank_with_leading_zeros>.log
     * @param debugLvl debug level
     */
    void init(MPI_Comm comm, const std::string fname, int debugLvl = 3)
    {
        MPI_Comm_rank(comm, &rank);
        std::string rankStr = std::string(5 - std::to_string(rank).length(), '0') + std::to_string(rank);

        auto pos = fname.find_last_of('.');
        auto start = fname.substr(0, pos);
        auto end = fname.substr(pos);

        fout = fopen( (start+"_"+rankStr+end).c_str(), "w");

        setDebugLvl(debugLvl);

        register_signals();
    }

    /**
     * Set logger to write to a certain, already opened file.
     *
     * @param comm relevant MPI communicator, typically \e MPI_COMM_WORLD
     * @param fout file handler, must be opened, typically \e stdout or \e stderr
     * @param debugLvl debug level
     */
    void init(MPI_Comm comm, FILE* fout, int debugLvl = 3)
    {
        MPI_Comm_rank(comm, &rank);
        this->fout = fout;

        setDebugLvl(debugLvl);
    }

  
    /**
     * Main logging function.
     * First, check message importance against debug level and return
     * if importance is too low (bigger number means lower importance)
     *
     * Then, construct a logger entry with time prefix, importance string,
     * filename and line number, and finally the message itself.
     *
     * And finally print output it to the file.
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
     * @tparam importance message importance
     * @param fname name of the current source file
     * @param lnum  line number of the source file
     * @param pattern message pattern to be passed to \e printf
     * @param args other relevant arguments to \e printf
     */
    template<int importance, class ... Args>
    inline void log(const char *fname, const int lnum, const char *pattern, Args... args) const
    {
        if (importance > runtimeDebugLvl) return;

        if (fout == nullptr)
        {
            fprintf(stderr, "Logger file is not set but tried to be used at %s : %d with the following message:\n", fname, lnum);
            fprintf(stderr, pattern, args...);
            fprintf(stderr, "\n");
            exit(1);
        }

        using namespace std::chrono;

        auto now   = system_clock::now();
        auto now_c = system_clock::to_time_t(now);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

        std::ostringstream tmout;
        tmout << std::put_time(std::localtime(&now_c), "%T") << ':' << std::setfill('0') << std::setw(3) << ms.count();

        const int cappedLvl = std::min((int)lvl2text.size() - 1, importance);
        std::string intro = tmout.str() + "   " + std::string("Rank %04d %7s at ")
            + fname + ":" + std::to_string(lnum) + "  " +pattern + "\n";

        FILE* ftmp = (fout != nullptr) ? fout : stdout;
        fprintf(ftmp, intro.c_str(), rank, (cappedLvl >= 0 ? lvl2text[cappedLvl] : "").c_str(), args...);


        bool needToFlush = runtimeDebugLvl >= flushThreshold && COMPILE_DEBUG_LVL >= flushThreshold;
        needToFlush = needToFlush || (now - lastFlushed > flushPeriod);
        if (needToFlush)
        {
            fflush(fout);
            lastFlushed = now;
        }
    }

    /**
     * Create a short error message for throwing an exception
     *
     */
    template<class ... Args>
    std::string makeSimpleErrString(const char* fname, const int lnum, const char* pattern, Args... args) const
    {
        int size = 10000;
        char buffer[size];
        
        std::string intro = "%s" + std::string(pattern); // shut up compiler warning
        snprintf(buffer, size, intro.c_str(), "", args...);
        
        return std::string(buffer);
    }

    /**
     * Special wrapper around log() that kills the application on
     * a fatal error
     * Print stack trace, error message, close the file and abort
     *
     * @param args forwarded to log()
     */
    template<class ... Args>
    inline void _die(Args ... args)
    {
        log<0>(args...);
        
        // print stacktrace
        std::ostringstream strace;
        pretty_stacktrace(strace);
        fwrite(strace.str().c_str(), sizeof(char), strace.str().size(), fout);

        fflush(fout);
        fclose(fout);
        fout = nullptr;

        throw std::runtime_error("Mirheo has encountered a fatal error and will quit now.\n"
                                 "The error message follows, and more details can be found in the log\n"
                                 "***************************************\n"
                                 "\t" + makeSimpleErrString(args...) + "\n"
                                 "***************************************");

    }

    /**
     * Check the return code of an MPI function. Default error checking of MPI
     * should be turned off
     *
     * @param fname name of the current source file
     * @param lnum  line number of the source file
     * @param code  error code (returned by MPI call)
     */
    inline void _MPI_Check(const char* fname, const int lnum, const int code)
    {
        if (code != MPI_SUCCESS)
        {
            char buf[MPI_MAX_ERROR_STRING];
            int nchar;
            MPI_Error_string(code, buf, &nchar);

            _die(fname, lnum, buf);
        }
    }

    int getDebugLvl()
    {
        return runtimeDebugLvl;
    }

    /// set debug level between 0 and #COMPILE_DEBUG_LVL
    void setDebugLvl(int debugLvl)
    {
        runtimeDebugLvl = std::max(std::min(debugLvl, COMPILE_DEBUG_LVL), 0);
        log<1>(__FILE__, __LINE__, "Compiled with maximum debug level %d", COMPILE_DEBUG_LVL);
        log<1>(__FILE__, __LINE__, "Debug level requested %d, set to %d", debugLvl, runtimeDebugLvl);
    }

    /**
     * @param fname name of the current source file
     * @param lnum  line number of the source file
     * @param code  error code (returned by CUDA call)
     */
    inline void _CUDA_Check(const char* fname, const int lnum, cudaError_t code)
    {
        if (code != cudaSuccess)
            _die(fname, lnum, cudaGetErrorString(code));
    }

private:
    int runtimeDebugLvl;           ///< debug level defined at runtime through setDebugLvl
    const int flushThreshold = 8;  ///< value of debug level starting with which every message
                                   ///< will be flushed to disk immediately

    mutable std::chrono::system_clock::time_point lastFlushed;
    const std::chrono::seconds flushPeriod{2};

    FILE* fout = nullptr;
    int rank;

    /**
     * Messages will be prefixed with the line from this array depending on their importance level
     */
    const std::array<std::string, 5> lvl2text{ {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG"} };
};

/// Unconditionally print to log, debug level is not checked here
#define   say(...)  logger.log<1>    (__FILE__, __LINE__, ##__VA_ARGS__)

#if COMPILE_DEBUG_LVL >= 0
/// Report a fatal error and abort
#define   die(...)  logger._die      (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define   die(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 1
/// Report a serious error
#define error(...)  logger.log<1>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define error(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 2
/// Report a warning
#define  warn(...)  logger.log<2>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define  warn(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 3
/// Report certain valuable information
#define  info(...)  logger.log<3>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define  info(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 4
/// Print debug output
#define debug(...)  logger.log<4>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 5
/// Print more debug
#define debug2(...) logger.log<5>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug2(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 6
/// Print yet more debug
#define debug3(...) logger.log<6>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug3(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 7
/// Print ultimately verbose debug. God help you scrambling through all the output
#define debug4(...) logger.log<7>    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug4(...)  do { } while(0)
#endif

/// Check an MPI call, call #die() if it fails
#define  MPI_Check(command) logger._MPI_Check (__FILE__, __LINE__, command)

/// Check a CUDA call, call #die() if it fails
#define CUDA_Check(command) logger._CUDA_Check(__FILE__, __LINE__, command)

/**
 * Inform all the object files that there is one Logger defined somewhere,
 * that they will be using to log stuff
 */
extern Logger logger;

