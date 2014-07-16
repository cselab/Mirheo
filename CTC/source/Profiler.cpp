/*
 *  Profiler.cpp
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 30.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <sstream>
#include <iomanip>

#include "Profiler.h"

inline void synchro()
{
#ifdef __MD_USE_CUDA__
	cudaThreadSynchronize();
#endif
}

Profiler::Profiler()
{
	mode = MSEC;
}

void Profiler::start(string name)
{
	Timings* tm;
	if (timings.find(name) == timings.end())
	{
		tm = &timings[name];
		tm->started    = true;
		tm->total      = 0;
		tm->iterations = 0;
	}
	else
	{
		tm = &timings[name];
		tm->started = true;
	}
	
	tm->start = mach_absolute_time();
	ongoing = name;
}

void Profiler::stop(string name)
{
	synchro();
	if (timings.find(name) != timings.end())
	{
		Timings* tm = &timings[name];
		if (tm->started)
		{
			tm->started = false;
			tm->total += mach_absolute_time() - tm->start;
			tm->iterations++;
		}
	}
}

void Profiler::stop()
{
	stop(ongoing);
}

double Profiler::elapsed(string name)
{
	if (timings.find(name) != timings.end())
	{
		Timings *tm = &timings[name];
		double res = tm->total / tm->iterations;
		tm->total = 0;
		tm->iterations = 0;
		return res;
	}
	return 0;
}

string Profiler::printStat()
{
	double total = 0;
	int longest = 0;
	ostringstream out;
	map<string, Timings>::iterator it;
	double now = mach_absolute_time();
	for (it = timings.begin(); it != timings.end(); it++)
	{
		if (it->second.started)
		{
			it->second.started = false;
			it->second.total += now - it->second.start;
		}
		
		total += it->second.total / it->second.iterations;
		if (longest < it->first.length())
			longest = it->first.length();
	}

	double factor = 1e-6;
	string unit;
	if (mode == SEC) 
	{
		factor = 1e-9;
		unit = "sec";
	}
	if (mode == MSEC)
	{
		factor = 1e-6;
		unit = "millisec";
	}
	if (mode == MCSEC)
	{
		factor = 1e-3;
		unit = "microsec";
	}
	longest = max(longest, 6);
	
	out << "Average total time: " << total*factor << " " << unit << endl;
	out << left << "[" << setw(longest) << "Kernel" << "]    " << setw(20) << "Time, "+unit << setw(20) << "Percentage" << endl;
	for (it = timings.begin(); it != timings.end(); it++)
	{
		out << "[" << setw(longest) << it->first << "]    "
			<< fixed << setprecision(3) << setw(20) << it->second.total * factor / it->second.iterations
			<< fixed << setprecision(1) << setw(20) << it->second.total / total * 100 / it->second.iterations << endl;
	}
	
	return out.str();
}


