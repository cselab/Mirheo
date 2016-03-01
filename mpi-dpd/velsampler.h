/*
 *  velsampler.h
 *  ctc PANDA
 *
 *  Created by Dmitry Alexeev on Nov 27, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>
#include "common.h"

using namespace std;

class VelSampler
{
public:
    struct CellInfo
    {
        uint cellsx, cellsy, cellsz;
    };

private:
    CellInfo info;
    int size, rank;

    SimpleDeviceBuffer<float3> vels;
    vector<float3> hostVels;
    int total, globtot;

    float3 desired;
    float Kp, Ki, Kd, factor;
    float3 s, old;
    float3 f;

    int sampleid;

public:
    VelSampler();

    void sample(const int * const cellsstart, const Particle* const p, cudaStream_t stream);
    vector<float3>& getAvgVel(cudaStream_t stream);
};



