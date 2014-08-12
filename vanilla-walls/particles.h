/*
 * Particles.h
 *
 *  Created on: Aug 12, 2014
 *      Author: kirill
 */
#pragma once

#include <cmath>

#include <vector>
#include <string>

#include "funnel-obstacle.h"

inline float saru(unsigned int seed1, unsigned int seed2, unsigned int seed3)
{
    seed3 ^= (seed1<<7)^(seed2>>6);
    seed2 += (seed1>>4)^(seed3>>15);
    seed1 ^= (seed2<<9)+(seed3<<8);
    seed3 ^= 0xA5366B4D*((seed2>>11) ^ (seed1<<1));
    seed2 += 0x72BE1579*((seed1<<4)  ^ (seed3>>16));
    seed1 ^= 0X3F38A6ED*((seed3>>5)  ^ (((signed int)seed2)>>22));
    seed2 += seed1*seed3;
    seed1 += seed3 ^ (seed2>>2);
    seed2 ^= ((signed int)seed2)>>17;

    int state  = 0x79dedea3*(seed1^(((signed int)seed1)>>14));
    int wstate = (state + seed2) ^ (((signed int)state)>>8);
    state  = state + (wstate*(wstate^0xdddf97f5));
    wstate = 0xABCB96F7 + (wstate>>1);

    state  = 0x4beb5d59*state + 0x2600e1f7; // LCG
    wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS

    unsigned int v = (state ^ (state>>26))+wstate;
    unsigned int r = (v^(v>>20))*0x6957f5a7;

    float res = r / (4294967295.0f);
    return res;
}

using namespace std;

struct Bouncer;

struct Particles
{
    float L[3];

    static int idglobal;

    int n, myidstart, steps_per_dump = 100;
    mutable int saru_tag;
    float xg = 0, yg = 0, zg = 0;
    vector<float> xp, yp, zp, xv, yv, zv, xa, ya, za;

    string name;

    Particles (const int n, const float Lx);
    Particles (const int n, const float Lx, const float Ly, const float Lz);

    void _dpd_forces_bipartite(const float kBT, const double dt,
                   const float * const srcxp, const float * const srcyp, const float * const srczp,
                   const float * const srcxv, const float * const srcyv, const float * const srczv,
                   const int nsrc,
                   const int giddstart, const int gidsstart);

    void acquire_global_id();


    void diag(FILE * f, float t);

    void vmd_xyz(const char * path, bool append = false);

    // might be opened by OVITO and xmovie (xwindow-based utility)
    void lammps_dump(const char* path, size_t timestep);

    void _dpd_forces(const float kBT, const double dt);
    void equilibrate(const float kBT, const double tend, const double dt, const Bouncer* bouncer);
};
