/*
 *  Misc.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cstdlib>
#include <cmath>
#include <time.h>

#pragma once

//**********************************************************************************************************************
// Randomer
//
// erand48() is used instead of drand48() to make it thread-safe
// just in case
//**********************************************************************************************************************
class Randomer
{
	unsigned short state[3];
	double prev;
	bool generated;
	
public:
	Randomer(int baseSeed = -1, int shiftSeed = 0)
	{
		generated = false;
		if (baseSeed == -1)
			baseSeed = time(NULL);
		
		unsigned short seed = (unsigned short)(baseSeed + shiftSeed);
		state[0] = state[1] = 0;
		state[2] = seed;
	}
	
	inline double getRand()
	{
		return erand48(state);
	}
	
	inline double getNormalRand()
	{
		if (generated)
		{
			generated = false;
			return prev;
		}
		
		generated  = true;
		double u1  = getRand();
		double u2  = getRand();
		
		double ln  = sqrt(-2.0 * log2(u1));
		double phi = 2 * M_PI * u2;
		prev       = ln * sin(phi);
		
		return ln * cos(phi);
	}
};







