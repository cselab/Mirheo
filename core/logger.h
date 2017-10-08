#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <execinfo.h>
#include <unistd.h>
#include <cstdio>
#include <string>

#include "stacktrace.h"

#include <mpi.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

#ifndef COMPILE_DEBUG_LVL
#define COMPILE_DEBUG_LVL 10
#endif

class Logger
{
	int runtimeDebugLvl;
	const int flushThreshold = 7;

	FILE* fout = nullptr;
	int rank;

	const std::array<std::string, 5> lvl2text{ {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG"} };

	template<int level, class ... Args>
	inline void log(const char* fname, const int lnum, const char* pattern, Args... args)
	{
		if (level <= runtimeDebugLvl)
		{
			if (fout == nullptr)
			{
				fprintf(stderr, "Logger file is not set\n");
				exit(1);
			}

			using namespace std::chrono;

			auto now   = system_clock::now();
			auto now_c = system_clock::to_time_t(now);
			auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

			std::ostringstream tmout;
			tmout << std::put_time(std::localtime(&now_c), "%T") << ':' << std::setfill('0') << std::setw(3) << ms.count();

			const int cappedLvl = std::min((int)lvl2text.size() - 1, level);
			std::string intro = tmout.str() + "   " + std::string("Rank %04d %7s at ")
				+ fname + ":" + std::to_string(lnum) + "  " +pattern + "\n";

			FILE* ftmp = (fout != nullptr) ? fout : stdout;
			fprintf(ftmp, intro.c_str(), rank, (cappedLvl >= 0 ? lvl2text[cappedLvl] : "").c_str(), args...);

			if (runtimeDebugLvl >= flushThreshold && COMPILE_DEBUG_LVL >= flushThreshold)
				fflush(fout);
		}
	}


public:

	Logger() : runtimeDebugLvl(-100) {}

	~Logger()
	{
		if (fout != nullptr)
		{
			fflush(fout);
			fclose(fout);
		}
	}

	void init(MPI_Comm&& comm, const std::string fname, int debugLvl = 3)
	{
		MPI_Comm_rank(comm, &rank);
		std::string rankStr = std::string(5 - std::to_string(rank).length(), '0') + std::to_string(rank);

		auto pos = fname.find_last_of('.');
		auto start = fname.substr(0, pos);
		auto end = fname.substr(pos);

		fout = fopen( (start+"_"+rankStr+end).c_str(), "w");

		setDebugLvl(debugLvl);
	}

	void init(MPI_Comm&& comm, FILE* fout, int debugLvl = 3)
	{
		MPI_Comm_rank(comm, &rank);
		this->fout = fout;

		setDebugLvl(debugLvl);
	}


	template<class ... Args>
	inline void _say(Args ... args)
	{
		log<-10>(args...);
	}

	template<class ... Args>
	inline void _die(Args ... args)
	{
		log<0>(args...);
		
		fflush(fout);
		fclose(fout);
		fout = nullptr;

		using namespace backward;
		StackTrace st;
		st.load_here(32);
		Printer p;
		p.object = true;
		p.color_mode = ColorMode::automatic;
		p.address = true;
		p.print(st, stderr);
		fflush(stderr);

		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	template<class ... Args>
	inline void _error(Args ... args)
	{
		log<1>(args...);
	}

	template<class ... Args>
	inline void _warn(Args ... args)
	{
		log<2>(args...);
	}

	template<class ... Args>
	inline void _info(Args ... args)
	{
		log<3>(args...);
	}

	template<class ... Args>
	inline void _debug(Args ... args)
	{
		log<4>(args...);
	}

	template<unsigned int EXTRA, class ... Args>
	inline void _debugX(Args ... args)
	{
		log<EXTRA+4>(args...);
	}

	inline void _MPI_Check(const char* fname, const int lnum, const int code)
	{
		if (code != MPI_SUCCESS)
		{
			char buf[2000];
			int nchar;
			MPI_Error_string(code, buf, &nchar);

			_die(fname, lnum, buf);
		}
	}

	int getDebugLvl()
	{
		return runtimeDebugLvl;
	}

	void setDebugLvl(int debugLvl)
	{
		runtimeDebugLvl = max(min(debugLvl, COMPILE_DEBUG_LVL), 0);
		_say(__FILE__, __LINE__, "Compiled with maximum debug level %d", COMPILE_DEBUG_LVL);
		_say(__FILE__, __LINE__, "Debug level requested %d, set to %d", debugLvl, runtimeDebugLvl);
	}


#ifdef __CUDACC__
	inline void _CUDA_Check(const char* fname, const int lnum, cudaError_t code)
	{
		if (code != cudaSuccess)
			_die(fname, lnum, cudaGetErrorString(code));
	}
#endif
};

extern Logger logger;

#define   say(...)  logger._say      (__FILE__, __LINE__, ##__VA_ARGS__)

#if COMPILE_DEBUG_LVL >= 0
#define   die(...)  logger._die      (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define   die(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 1
#define error(...)  logger._error    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define error(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 2
#define  warn(...)  logger._warn     (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define  warn(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 3
#define  info(...)  logger._info     (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define  info(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 4
#define debug(...)  logger._debug    (__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 5
#define debug2(...) logger._debugX<1>(__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug2(...)  do { } while(0)
#endif

#if COMPILE_DEBUG_LVL >= 5
#define debug3(...) logger._debugX<2>(__FILE__, __LINE__, ##__VA_ARGS__)
#else
#define debug3(...)  do { } while(0)
#endif

#define  MPI_Check(command) logger._MPI_Check (__FILE__, __LINE__, command)
#define CUDA_Check(command) logger._CUDA_Check(__FILE__, __LINE__, command)



