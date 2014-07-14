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

using namespace std;

//**********************************************************************************************************************
// DPD
//**********************************************************************************************************************
class DPD
{
private:
	double alpha, gamma, temp, sigma, dt_1;
    double rCut2;
    
    mutable mt19937 gen;
    mutable normal_distribution<double> norm;
    
    inline double w(double) const;
    
public:
	double rCut;
	inline DPD(double, double, double, double, double, int);
    __host__ __device__ inline void   F(const double, const double, const double, const double, const double, const double, double&, double&, double&) const;
};

inline DPD::DPD(double alpha, double gamma, double temp, double rCut, double dt, int seed = 0):
alpha(alpha), gamma(gamma), temp(temp), rCut(rCut), rCut2(rCut * rCut),
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp / dt)), dt_1(1.0/sqrt(dt))
{
}

inline double DPD::w(double r) const
{
    //TODO implement with masking operations
    return (r<rCut) ? 1 - r/rCut : 0;
}

__host__ __device__ inline void DPD::F(const double dx, const double dy, const double dz,
                                       const double vx, const double vy, const double vz,
                                       double& fx,      double& fy,      double& fz) const
{
	const double r2 = dx*dx + dy*dy + dz*dz;
	if (r2 > rCut2)
	{
		fx = 0;
		fy = 0;
		fz = 0;
		return;
	}
	
    double IrI = sqrt(r2);
    double wr_IrI = w(IrI) / IrI;
	   
    double fc = alpha * wr_IrI;
    double fd = -gamma * wr_IrI * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    double fr = sigma * wr_IrI * norm(gen);
    
    double fAbs = -(fc + fd + fr);
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
	double gamma, temp, sigma, dt_1;
    double u0, rho, req;
    double rCut2, req2;
    
    mutable mt19937 gen;
    mutable normal_distribution<double> norm;
    
    inline double w(double) const;
    
public:
	double rCut;
	inline SEM(double, double, double, double, double, double, double, int);
    __host__ __device__ inline void   F(const double, const double, const double, const double, const double, const double, double&, double&, double&) const;
};

inline SEM::SEM(double gamma, double temp, double rCut, double dt,   double u0, double rho, double req, int seed = 0):
gamma(gamma), temp(temp), rCut(rCut), rCut2(rCut * rCut),
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp)), dt_1(1.0/sqrt(dt)),
u0(u0), rho(rho), req(req), req2(req*req)
{
}

inline double SEM::w(double r) const
{
    //TODO implement with masking operations
    return (r<rCut) ? 1 - r/rCut : 0;
}

__host__ __device__ inline void SEM::F(const double dx, const double dy, const double dz,
                                       const double vx, const double vy, const double vz,
                                       double& fx,      double& fy,      double& fz) const
{
	const double r2 = dx*dx + dy*dy + dz*dz;
	if (r2 > rCut2)
	{
		fx = 0;
		fy = 0;
		fz = 0;
		return;
	}
	
    double IrI = sqrt(r2);
    double wr_IrI = w(IrI) / IrI;
        
    double exponent = exp(rho * (1 - r2/req2));
    double fc = -2*u0 * exponent * (1 - exponent);
    double fd = -gamma * wr_IrI * wr_IrI * (dx*vx + dy*vy + dz*vz);   // !!! minus
    double fr = sigma * wr_IrI * norm(gen) * dt_1;
    
    double fAbs = -(fc + fd + fr);
    fx = fAbs * dx;
	fy = fAbs * dy;
	fz = fAbs * dz;
}

template<int a, int b>
void force(const double rx, const double ry, const double rz,
           const double vx, const double vy, const double vz,
           double& fx,      double& fy,      double& fz)
{
}

template<>
void force<0, 0>(const double rx, const double ry, const double rz,
                   const double vx, const double vy, const double vz,
                   double& fx,      double& fy,      double& fz)
{
    static const double aij   = configParser->getf("DPD-DPD", "aij",    2.5);
    static const double gamma = configParser->getf("DPD-DPD", "gamma",  45);
    static const double temp  = configParser->getf("Basic",   "temp",   0.1);
    static const double rCut  = configParser->getf("DPD-DPD", "rCut",   1);
    static const double dt    = configParser->getf("Basic",   "dt",     0.001);
    
    static const DPD dpd(aij, gamma, temp, rCut, dt);
    
    dpd.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<0, 1>(const double rx, const double ry, const double rz,
                   const double vx, const double vy, const double vz,
                   double& fx,      double& fy,      double& fz)
{
    static const double aij   = configParser->getf("DPD-SEM", "aij",    3.5);
    static const double gamma = configParser->getf("DPD-SEM", "gamma",  45);
    static const double temp  = configParser->getf("Basic",   "temp",   0.1);
    static const double rCut  = configParser->getf("DPD-SEM", "rCut",   1);
    static const double dt    = configParser->getf("Basic",   "dt",     0.001);

    static const DPD dpd(aij, gamma, temp, rCut, dt);

    dpd.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<1, 1>(const double rx, const double ry, const double rz,
                   const double vx, const double vy, const double vz,
                   double& fx,      double& fy,      double& fz)
{
    static const double gamma = configParser->getf("SEM-SEM", "gamma",  600);
    static const double temp  = configParser->getf("Basic",   "temp",   0.1);
    static const double rCut  = configParser->getf("SEM-SEM", "rCut",   1);
    static const double dt    = configParser->getf("Basic",   "dt",     0.001);
    static const double u0    = configParser->getf("SEM-SEM", "u0",     0.14);
    static const double rho   = configParser->getf("SEM-SEM", "rho",    0.5);
    static const double req   = configParser->getf("SEM-SEM", "req",    0.76);


    static const SEM sem(gamma, temp, rCut, dt,  u0, rho, req);
    
    sem.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}



