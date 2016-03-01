/*
 *  velcontroller.h
 *  ctc falcon
 *
 *  Created by Dmitry Alexeev on Sep 24, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "common.h"


class VelController
{
public:
    struct CellInfo
    {
        uint cellsx, cellsy, cellsz;
        uint xl[3];
        uint n[3];
    };

private:
    CellInfo info;
    MPI_Comm comm;
    int size, rank;

    SimpleDeviceBuffer<float3> vel;
    PinnedHostBuffer<float3> avgvel;
    int total, globtot;

    float3 desired;
    float Kp, Ki, Kd, factor;
    float3 s, old;
    float3 f;

    int sampleid;

public:
    VelController(int xl[3], int xh[3], int mpicoos[3], float3 desired, MPI_Comm comm);

    void sample(const int * const cellsstart, const Particle* const p, cudaStream_t stream);
    void push  (const int * const cellsstart, const Particle* const p, Acceleration* acc, cudaStream_t stream);
    float3 adjustF(cudaStream_t stream);
};




