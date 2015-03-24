/*
 *  cell-destroyer.h
 *  ctc-garbald
 *
 *  Created by Dmitry Alexeev on Mar 22, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

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

class CellDestroyer
{
	float xlim;
	int xcoo;

	Logistic::KISS rng;

	inline float toGlobal(float x);

public:

	CellDestroyer(float xlim, int xcoo);
	int destroy(ParticleArray& p, CollectionRBC* rbcs);
};





