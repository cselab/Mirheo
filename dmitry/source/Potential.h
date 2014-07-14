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
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp)), dt_1(1.0/sqrt(dt))
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
    double IrI_1 = 1.0 / IrI;
	   
    double wr = w(IrI);
    double fc = alpha * wr * IrI_1;
    double fd = -gamma * wr * wr * (dx*vx + dy*vy + dz*vz) / r2;   // !!! minus
    double fr = sigma * wr * norm(gen) * IrI_1 * dt_1;
    
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
    double IrI_1 = 1.0 / IrI;
    
    double wr = w(IrI);
    
    double exponent = exp(rho * (1 - r2/req2));
    double fc = -2*u0 * exponent * (1 - exponent);
    double fd = -gamma * wr * wr * (dx*vx + dy*vy + dz*vz) / r2;   // !!! minus
    double fr = sigma * wr * norm(gen) * IrI_1 * dt_1;
    
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
    static const DPD dpd(2.5, 45, 0.1, 1, 1e-3);
    
    dpd.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<0, 1>(const double rx, const double ry, const double rz,
                   const double vx, const double vy, const double vz,
                   double& fx,      double& fy,      double& fz)
{
    static const DPD dpd(3.5, 45, 0.1, 1, 1e-3);

    dpd.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}

template<>
void force<1, 1>(const double rx, const double ry, const double rz,
                   const double vx, const double vy, const double vz,
                   double& fx,      double& fy,      double& fz)
{
    static const SEM sem(600, 0.1, 1, 1e-3,  0.14, 0.5, 0.76);
    
    sem.F(rx, ry, rz,  vx, vy, vz,  fx, fy, fz);
}



