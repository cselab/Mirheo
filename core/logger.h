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

#include <mpi.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

#ifndef DEBUGLVL
#define DEBUGLVL 9
#endif

class Logger
{
	int runtimeDebugLvl;

	FILE* fout = nullptr;
	int rank;

	const std::array<std::string, 5> lvl2text{ {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG"} };

	template<int level, class ... Args>
	inline void log(const char* fname, const int lnum, const char* pattern, Args... args)
	{
		if (level <= DEBUGLVL && level <= runtimeDebugLvl)
		{
			using namespace std::chrono;

			auto now   = system_clock::now();
			auto now_c = system_clock::to_time_t(now);
			auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

			std::ostringstream tmout;
			tmout << std::put_time(std::localtime(&now_c), "%T") << ':' << std::setfill('0') << std::setw(3) << ms.count();

			const int cappedLvl = std::min((int)lvl2text.size() - 1, level);
			std::string intro = tmout.str() + "   " + std::string("Rank %04d %7s at ")
				+ fname + ":" + std::to_string(lnum) + "  " +pattern + "\n";

			fprintf(fout, intro.c_str(), rank, (cappedLvl >= 0 ? lvl2text[cappedLvl] : "").c_str(), args...);
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
		runtimeDebugLvl = debugLvl;

		MPI_Comm_rank(comm, &rank);
		std::string rankStr = std::string(5 - std::to_string(rank).length(), '0') + std::to_string(rank);

		auto pos = fname.find_last_of('.');
		auto start = fname.substr(0, pos);
		auto end = fname.substr(pos);

		fout = fopen( (start+"_"+rankStr+end).c_str(), "w");
	}

	void init(MPI_Comm&& comm, FILE* fout, int debugLvl = 3)
	{
		runtimeDebugLvl = debugLvl;
		MPI_Comm_rank(comm, &rank);
		this->fout = fout;
	}


	template<class ... Args>
	inline void _say(Args ... args)
	{
		log<-1>(args...);
	}

	template<class ... Args>
	inline void _die(Args ... args)
	{
		log<0>(args...);
		
		fflush(fout);
		fclose(fout);
		fout = nullptr;

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

	int& debugLvl()
	{
		return runtimeDebugLvl;
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
#define   die(...)  logger._die      (__FILE__, __LINE__, ##__VA_ARGS__)
#define error(...)  logger._error    (__FILE__, __LINE__, ##__VA_ARGS__)
#define  warn(...)  logger._warn     (__FILE__, __LINE__, ##__VA_ARGS__)
#define  info(...)  logger._info     (__FILE__, __LINE__, ##__VA_ARGS__)
#define debug(...)  logger._debug    (__FILE__, __LINE__, ##__VA_ARGS__)
#define debug2(...) logger._debugX<1>(__FILE__, __LINE__, ##__VA_ARGS__)
#define debug3(...) logger._debugX<2>(__FILE__, __LINE__, ##__VA_ARGS__)

#define  MPI_Check(command) logger._MPI_Check (__FILE__, __LINE__, command)
#define CUDA_Check(command) logger._CUDA_Check(__FILE__, __LINE__, command)



