/*
 *  dumper.h
 *  ctc daint
 *
 *  Created by Dmitry Alexeev on Sep 24, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <mpi.h>
#include <vector>

#include "common.h"

using namespace std;

class Dumper
{
    MPI_Comm iocomm, intercomm, iocartcomm;
    vector<Particle> particles;
    vector<Acceleration> accelerations;

    int nrbcverts, nctcverts, qoiid, rank;

    void qoi(Particle* rbcs, Particle * ctcs, int nrbcparts, int nctcparts, const float tm);

public:
    Dumper(MPI_Comm iocomm, MPI_Comm iocartcomm, MPI_Comm intercomm);
    void do_dump();
};

