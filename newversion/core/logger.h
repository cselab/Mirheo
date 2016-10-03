#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <array>
#include <algorithm>

#include <mpi.h>

#ifdef __CUDACC__
#include <cuda.h>
#endif

#ifndef DEBUGLVL
#define DEBUGLVL 3
#endif

class Logger
{
	int runtimeDebugLvl;

	MPI_File fout;
	int rank;

	const std::array<std::string, 5> lvl2text{ {"FATAL", "ERROR", "WARNING", "INFO", "DEBUG"} };

	template<int level, class ... Args>
	inline void log(const char* pattern, Args... args)
	{
		if (level <= DEBUGLVL && level <= runtimeDebugLvl)
		{
			const int cappedLvl = std::min((int)lvl2text.size() - 1, level);
			std::string intro = std::string("Rank %03d %7s at ") + __FILE__ + ":" + std::to_string(__LINE__) + "  " +pattern + "\n";

			MPI_Status status;
			char buf[2000];
			int nchar = sprintf(buf, intro.c_str(), rank, (cappedLvl >= 0 ? lvl2text[cappedLvl] : "").c_str(), args...);
			MPI_File_write_shared(fout, buf, nchar, MPI_CHAR, &status);
		}
	}

public:

	void init(MPI_Comm&& comm, const std::string fname, int debugLvl = DEBUGLVL)
	{
		runtimeDebugLvl = debugLvl;

		MPI_Info infoin;
		MPI_Info_create(&infoin);
		MPI_Info_set(infoin, "access_style", "write_once,random");
		MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, infoin, &fout);
		MPI_File_set_atomicity(fout, true);
		MPI_Comm_rank(comm, &rank);
	}


	template<class ... Args>
	inline void say(const char* pattern, Args ... args)
	{
		log<-1>(pattern, args...);
	}

	template<class ... Args>
	inline void die(const char* pattern, Args ... args)
	{
		log<0>(pattern, args...);
		MPI_File_close(&fout);
		MPI_Abort(MPI_COMM_WORLD, -1);
	}

	template<class ... Args>
	inline void error(const char* pattern, Args ... args)
	{
		log<1>(pattern, args...);
	}

	template<class ... Args>
	inline void warn(const char* pattern, Args ... args)
	{
		log<2>(pattern, args...);
	}

	template<class ... Args>
	inline void info(const char* pattern, Args ... args)
	{
		log<3>(pattern, args...);
	}

	template<class ... Args>
	inline void debug(const char* pattern, Args ... args)
	{
		log<4>(pattern, args...);
	}

	void MPI_Check(const int code)
	{
		if (code != MPI_SUCCESS)
		{
			char buf[2000];
			int nchar;
			MPI_Error_string(code, buf, &nchar);

			log<0>(buf);
			MPI_File_close(&fout);
			MPI_Abort(MPI_COMM_WORLD, code);
		}
	}

	int& debugLvl()
	{
		return runtimeDebugLvl;
	}


#ifdef __CUDACC__
	void CUDA_Check(cudaError_t code)
	{
		if (code != cudaSuccess)
		{
			die(cudaGetErrorString(code));
		}
	}
#endif
};

extern Logger logger;

