/*
 *  Potential.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 16.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "thrust/detail/config.h"
#include <random>

#include "Misc.h"
#include "ErrorHandling.h"

using namespace std;
using namespace ErrorHandling;

//**********************************************************************************************************************
// DPD
//**********************************************************************************************************************
class DPD
{
private:
	real alpha, gamma, temp, sigma, dt_1;
    real rCut2;
    
    mutable mt19937 gen;
    mutable normal_distribution<real> norm;
    
    inline real w(real) const;
    
public:
	real rCut;
	inline DPD(real, real, real, real, real, int);
    __host__ __device__ inline void   F(const real, const real, const real, const real, const real, const real, real&, real&, real&) const;
};

inline DPD::DPD(real alpha, real gamma, real temp, real rCut, real dt, int seed = 0):
alpha(alpha), gamma(gamma), temp(temp), rCut(rCut), rCut2(rCut * rCut),
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp / dt)), dt_1(1.0/sqrt(dt))
{
}

inline real DPD::w(real r) const
{
    //TODO implement with masking operations
    return (r<rCut) ? 1 - r/rCut : 0;
}

__host__ __device__ inline void DPD::F(const real dx, const real dy, const real dz,
                                       const real vx, const real vy, const real vz,
                                       real& fx,      real& fy,      real& fz) const
{
	const real r2 = dx*dx + dy*dy + dz*dz;
	if (r2 > rCut2)
	{
		fx = 0;
		fy = 0;
		fz = 0;
		return;
	}
	
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
class SEM
{
private:
	real gamma, temp, sigma, dt_1;
    real u0, rho, req;
    real rCut2, req2;
    real D;
    
    mutable mt19937 gen;
    mutable normal_distribution<real> norm;
    
    inline real w(real) const;
    
public:
	real rCut;
	inline SEM(real, real, real, real, real, real, real, real, int);
    __host__ __device__ inline void   F(const real, const real, const real, const real, const real, const real, real&, real&, real&) const;
};

inline SEM::SEM(real gamma, real temp, real rCut, real dt,   real u0, real rho, real req, real D, int seed = 0):
gamma(gamma), temp(temp), rCut(rCut), rCut2(rCut * rCut),
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp)), dt_1(1.0/sqrt(dt)),
u0(u0), rho(rho), req(req), req2(req*req), D(D)
{
}

inline real SEM::w(real r) const
{
    //TODO implement with masking operations
    return (r<rCut) ? 1 - r/rCut : 0;
}

__host__ __device__ inline void SEM::F(const real dx, const real dy, const real dz,
                                       const real vx, const real vy, const real vz,
                                       real& fx,      real& fy,      real& fz) const
{
	const real r2 = dx*dx + dy*dy + dz*dz;
	
    real IrI = sqrt(r2);
    real wr_IrI = w(IrI) / IrI;
        
    real exponent = exp(rho * (1 - r2/req2));
    real fc = -2*u0 * exponent * (1 - exponent);
    real fd = -gamma * wr_IrI * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    real fr = sigma * D * wr_IrI * norm(gen) * dt_1;
    
    real fAbs = -(fc + fd + fr);
    
    debug("%f     %f\n", IrI, fAbs);
    
    fx = fAbs * dx;
	fy = fAbs * dy;
	fz = fAbs * dz;
}

template<int a, int b>
void force(const real rx, const real ry, const real rz,
           const real vx, const real vy, const real vz,
           real& fx,      real& fy,      real& fz)
{
}

template<>
void force<0, 0>(const real rx, const real ry, const real rz,
                   const real vx, const real vy, const real vz,
                   real& fx,      real& fy,      real& fz)
{
    static const real aij   = configParser->getf("DPD-DPD", "aij",    2.5);
    static const real gamma = configParser->getf("DPD-DPD", "gamma",  45);
    static const real temp  = configParser->getf("Basic",   "temp",   0.1);
    static const real rCut  = configParser->getf("DPD-DPD", "rCut",   1);
    static const real dt    = configParser->getf("Basic",   "dt",     0.001);
    
    static const DPD dpd(aij, gamma, temp, rCut, dt);
    
    dpd.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<0, 1>(const real rx, const real ry, const real rz,
                   const real vx, const real vy, const real vz,
                   real& fx,      real& fy,      real& fz)
{
    static const real aij   = configParser->getf("DPD-SEM", "aij",    3.5);
    static const real gamma = configParser->getf("DPD-SEM", "gamma",  45);
    static const real temp  = configParser->getf("Basic",   "temp",   0.1);
    static const real rCut  = configParser->getf("DPD-SEM", "rCut",   1);
    static const real dt    = configParser->getf("Basic",   "dt",     0.001);

    static const DPD dpd(aij, gamma, temp, rCut, dt);

    dpd.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<1, 1>(const real rx, const real ry, const real rz,
                   const real vx, const real vy, const real vz,
                   real& fx,      real& fy,      real& fz)
{
    static const real gamma = configParser->getf("SEM-SEM", "gamma",  600);
    static const real temp  = configParser->getf("Basic",   "temp",   0.1);
    static const real rCut  = configParser->getf("SEM-SEM", "rCut",   1);
    static const real dt    = configParser->getf("Basic",   "dt",     0.001);
    static const real u0    = configParser->getf("SEM-SEM", "u0",     0.14);
    static const real rho   = configParser->getf("SEM-SEM", "rho",    0.5);
    static const real req   = configParser->getf("SEM-SEM", "req",    0.76);
    static const real D     = configParser->getf("SEM-SEM", "D",      10e5);

    static const SEM sem(gamma, temp, rCut, dt,  u0, rho, req, D);
    
    sem.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<1, 2>(const real rx, const real ry, const real rz,
                 const real vx, const real vy, const real vz,
                 real& fx,      real& fy,      real& fz)
{
    force<1,1>(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<2, 2>(const real rx, const real ry, const real rz,
                 const real vx, const real vy, const real vz,
                 real& fx,      real& fy,      real& fz)
{
    force<1,1>(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}



