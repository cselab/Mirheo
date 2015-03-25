/*
 *  cell-producer.h
 *  ctc-garbald
 *
 *  Created by Dmitry Alexeev on Mar 21, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <vector>
#include <map>

#include <cuda_profiler_api.h>

#include "common.h"
#include "containers.h"
#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "rbc-interactions.h"
#include "ctc.h"

#pragma once

class CellProducer
{
	int n, m;
	int **occupied;
	float xmin, xmax;
	float hy, hz;
	int xcoo;

	Logistic::KISS rng;

	inline float toGlobal(float x);

public:

	CellProducer(int n, int m, float xmin, float xmax, int xcoo);
	bool produce(ParticleArray& p, CollectionRBC* rbcs);
};


