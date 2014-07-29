/*
 *  Interaction.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 16.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <random>

#include "Misc.h"
#include "ErrorHandling.h"
#include "Profiler.h"
#include "Particles.h"

using namespace std;
using namespace ErrorHandling;

template<typename Object>
class Cells;

class Interaction
{
    // i'm an invisible man
    
protected:
    inline static real saru(int seed1, int seed2, int seed3)
    {
        seed3 ^= (seed1<<7)^(seed2>>6);
        seed2 += (seed1>>4)^(seed3>>15);
        seed1 ^= (seed2<<
                  9)+(seed3<<8);
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
        
        state  = 0x4beb5d59*state + 0x2600e1f7;             // LCG
        wstate = wstate + 0x8009d14b + ((((signed int)wstate)>>31)&0xda879add); // OWS
        
        unsigned int v = (state ^ (state>>26))+wstate;
        unsigned int r = (v^(v>>20))*0x6957f5a7;
        
        real res = r / ((real)4294967295.0);
        return (real)3.464101615 * res - (real)1.732050807;
    }
};

//**********************************************************************************************************************
// DPD
//**********************************************************************************************************************
class DPD : public Interaction
{
private:
	real alpha, gamma, temp, sigma;
    real rCut, rCut2, rc;
    
    inline real w(real) const;
    
public:
    
    DPD(real rCut, real alpha, real gamma, real temp, real rc, real dt);
    
    inline bool  nonzero(real r2) { return r2 < rCut2; }
    inline void  F(const real dx, const real dy, const real dz,
                   const real vx, const real vy, const real vz,
                   const real r2,
                   real& fx,      real& fy,      real& fz,
                   int i, int j, int t) const;
};

inline DPD::DPD(real rCut, real alpha, real gamma, real temp, real rc, real dt):
    rCut(rCut), rCut2(rCut * rCut),
    alpha(alpha), gamma(gamma), temp(temp), rc(rc),
    sigma(sqrt(2 * gamma * temp / dt))
{
}

inline real DPD::w(real r) const
{
    // Cutoff is already checked here
    return 1 - r/rc;
}

inline void DPD::F(const real dx, const real dy, const real dz,
                   const real vx, const real vy, const real vz,
                   const real r2,
                   real& fx,      real& fy,      real& fz,
                   int i, int j, int t) const
{
    real IrI = sqrt(r2);
    real wr_IrI = w(IrI) / IrI;
	   
    //real fc = alpha;
    real fd = -gamma * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    real fr = sigma * Interaction::saru(min(i, j), max(i, j), t);
    
    real fAbs = -(alpha + fd + fr) * wr_IrI;
    fx = fAbs * dx;
	fy = fAbs * dy;
	fz = fAbs * dz;
}


//**********************************************************************************************************************
// SEM+DPD
//**********************************************************************************************************************
class SEM : public Interaction
{
private:
	real gamma, temp, sigma;
    real u0, rho, D;
    real rCut, rCut2, req_1, req_2, rc, rc2;
    
    inline real w(real) const;
    
public:
    SEM(real rCut,  real gamma, real temp, real rc, real dt,   real u0, real rho, real req, real D);
    
    inline bool nonzero(real r2) { return r2 < rCut2; }
    inline void  F(const real dx, const real dy, const real dz,
                   const real vx, const real vy, const real vz,
                   const real r2,
                   real& fx,      real& fy,      real& fz,
                   int i, int j, int t) const;
};

inline SEM::SEM(real rCut,  real gamma, real temp, real rc, real dt,   real u0, real rho, real req, real D):
    rCut(rCut), rCut2(rCut * rCut),
    gamma(gamma), temp(temp), rc(rc), sigma(sqrt(2 * gamma * temp / dt)),
    u0(u0), rho(rho), req_1(1.0/req), req_2(1.0/req/req), D(D)
{
}

inline real SEM::w(real r) const
{
    //Interaction cutoff != rc, need to check here
    return (r<rc) ? 1 - r/rc : 0;
}

inline void SEM::F(const real dx, const real dy, const real dz,
                   const real vx, const real vy, const real vz,
                   const real r2,
                   real& fx,      real& fy,      real& fz,
                   int i, int j, int t) const
{
    real IrI = sqrt(r2);
    real wr_IrI = w(IrI) / IrI;
        
    real exponent = exp(rho * (1 - r2*req_2));
    real fc = -4*u0*rho*req_2* exponent * (1 - exponent)/IrI;
    real fd = -gamma * wr_IrI * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    real fr = sigma * D * wr_IrI * Interaction::saru(min(i, j), max(i, j), t);
    
    real fAbs = -(fc + fd + fr);
    
    //debug2("%f     %f\n", IrI, fAbs);
    
    fx = fAbs * dx;
	fy = fAbs * dy;
	fz = fAbs * dz;
}


//**********************************************************************************************************************
// Interaction Table
//**********************************************************************************************************************
enum InterTypes { INTER_DPD, INTER_SEM };

class InteractionTable
{
    int nTypes;
    Particles**          part;
    real** rCut;
    Interaction*** table;
    InterTypes**   types;
    
    Profiler& prof;
    
public:
    Cells<Particles>*** cells;

    InteractionTable(int nTypes, Particles** part, Profiler& prof, real xlo, real xhi,  real ylo, real yhi,  real zlo, real zhi);
    void doCells();
    void evalForces(int step);
};


