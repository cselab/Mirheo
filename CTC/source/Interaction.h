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
#include "CellList.h"
#include "Profiler.h"
#include "Particles.h"

using namespace std;
using namespace ErrorHandling;

class Interaction
{
    // i'm the invisible man
};

//**********************************************************************************************************************
// DPD
//**********************************************************************************************************************
class DPD : public Interaction
{
private:
	real alpha, gamma, temp, sigma;
    real rCut, rCut2, rc;
    
    mutable mt19937 gen;
    mutable normal_distribution<real> norm;
    
    inline real w(real) const;
    
public:
    
    DPD(real rCut, real alpha, real gamma, real temp, real rc, real dt, int seed = 0);
    
    inline bool  nonzero(real r2) { return r2 < rCut2; }
    inline void  F(const real dx, const real dy, const real dz,
                   const real vx, const real vy, const real vz,
                   const real r2,
                   real& fx,      real& fy,      real& fz) const;
};

inline DPD::DPD(real rCut, real alpha, real gamma, real temp, real rc, real dt, int seed):
    rCut(rCut), rCut2(rCut * rCut),
    alpha(alpha), gamma(gamma), temp(temp), rc(rc),
    sigma(sqrt(2 * gamma * temp / dt)),
    gen(seed), norm(0, 1)
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
                   real& fx,      real& fy,      real& fz) const
{
    real IrI = sqrt(r2);
    real wr_IrI = w(IrI) / IrI;
	   
    //real fc = alpha;
    real fd = -gamma * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    real fr = sigma * norm(gen);
    
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
    
    mutable mt19937 gen;
    mutable normal_distribution<real> norm;
    
    inline real w(real) const;
    
public:
    SEM(real rCut,  real gamma, real temp, real rc, real dt,   real u0, real rho, real req, real D, int seed = 0);
    
    inline bool nonzero(real r2) { return r2 < rCut2; }
    inline void  F(const real dx, const real dy, const real dz,
                   const real vx, const real vy, const real vz,
                   const real r2,
                   real& fx,      real& fy,      real& fz) const;
};

inline SEM::SEM(real rCut,  real gamma, real temp, real rc, real dt,   real u0, real rho, real req, real D, int seed):
    rCut(rCut), rCut2(rCut * rCut),
    gamma(gamma), temp(temp), rc(rc), sigma(sqrt(2 * gamma * temp / dt)),
    u0(u0), rho(rho), req_1(1.0/req), req_2(1.0/req*req), D(D),
    gen(seed), norm(0, 1)
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
                   real& fx,      real& fy,      real& fz) const
{
    real IrI = sqrt(r2);
    real wr_IrI = w(IrI) / IrI;
        
    real exponent = exp(rho * (1 - r2*req_2));
    real fc = -4*u0*rho*req_1*req_1 * exponent * (1 - exponent)/IrI;
    real fd = -gamma * wr_IrI * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    real fr = sigma * D * wr_IrI * norm(gen);
    
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
    Cells<Particles>*** cells;
    Particles**          part;
    real** rCut;
    Interaction*** table;
    InterTypes**   types;
    
    Profiler& prof;
    
public:
    InteractionTable(int nTypes, Particles** part, Profiler& prof, real xlo, real xhi,  real ylo, real yhi,  real zlo, real zhi);
    void doCells();
    void evalForces();
};


