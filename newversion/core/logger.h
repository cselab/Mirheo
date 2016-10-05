#pragma once

#include <cstdlib>
#include <cstdio>
#include <string>
#include <array>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

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

	MPI_File fout;
	int rank;

	const std::array<std::string, 5> lvl2text{ {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG"} };

	template<int level, class ... Args>
	inline void log(const char* fname, const int lnum, const char* pattern, Args... args)
	{
		if (level <= DEBUGLVL && level <= runtimeDebugLvl)
		{
			auto now   = std::chrono::system_clock::now();
			auto now_c = std::chrono::system_clock::to_time_t(now);
			std::ostringstream tmout;
			tmout << std::put_time(std::localtime(&now_c), "%T");

			const int cappedLvl = std::min((int)lvl2text.size() - 1, level);
			std::string intro = tmout.str() + "   " + std::string("Rank %03d %7s at ")
				+ fname + ":" + std::to_string(lnum) + "  " +pattern + "\n";

			MPI_Status status;
			char buf[2000];
			int nchar = sprintf(buf, intro.c_str(), rank, (cappedLvl >= 0 ? lvl2text[cappedLvl] : "").c_str(), args...);
			MPI_File_write_shared(fout, buf, nchar, MPI_CHAR, &status);
		}
	}

public:

	void init(MPI_Comm&& comm, const std::string fname, int debugLvl = 3)
	{
		runtimeDebugLvl = debugLvl;

		MPI_Info infoin;
		MPI_Info_create(&infoin);
		MPI_Info_set(infoin, "access_style", "write_once,random");

		// If file exists - delete it
		MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE | MPI_MODE_DELETE_ON_CLOSE, infoin, &fout);
		MPI_File_close(&fout);

		MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, infoin, &fout);
		MPI_File_set_atomicity(fout, true);
		MPI_Comm_rank(comm, &rank);
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
		MPI_File_close(&fout);
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

	void _MPI_Check(const char* fname, const int lnum, const int code)
	{
		if (code != MPI_SUCCESS)
		{
			char buf[2000];
			int nchar;
			MPI_Error_string(code, buf, &nchar);

			log<0>(fname, lnum, buf);
			MPI_File_close(&fout);
			MPI_Abort(MPI_COMM_WORLD, code);
		}
	}

	int& debugLvl()
	{
		return runtimeDebugLvl;
	}


#ifdef __CUDACC__
	void _CUDA_Check(const char* fname, const int lnum, cudaError_t code)
	{
		if (code != cudaSuccess)
		{
			log<0>(fname, lnum, cudaGetErrorString(code));
			MPI_File_close(&fout);
			MPI_Abort(MPI_COMM_WORLD, code);
		}
	}
#endif
};

extern Logger logger;

#define   say(...) logger._say  (__FILE__, __LINE__, ##__VA_ARGS__)
#define   die(...) logger._die  (__FILE__, __LINE__, ##__VA_ARGS__)
#define error(...) logger._error(__FILE__, __LINE__, ##__VA_ARGS__)
#define  warn(...) logger._warn (__FILE__, __LINE__, ##__VA_ARGS__)
#define  info(...) logger._info (__FILE__, __LINE__, ##__VA_ARGS__)
#define debug(...) logger._debug(__FILE__, __LINE__, ##__VA_ARGS__)

#define  MPI_Check(command) logger._MPI_Check (__FILE__, __LINE__, command)
#define CUDA_Check(command) logger._CUDA_Check(__FILE__, __LINE__, command)



