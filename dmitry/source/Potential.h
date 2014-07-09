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
// Lennard-Jones
//**********************************************************************************************************************
class LennardJones
{
private:
	double epsx4;
	double epsx24;
	double sigma2;
	double rCut2;
	double cutoffShift;
		
public:
	double rCut;
	inline LennardJones(double, double, double);
	__host__ __device__ inline void   F(const double, const double, const double, double&, double&, double&);
	__host__ __device__ inline double V(const double, const double, const double);
	__host__ __device__ inline double _VnoCut(const double, const double, const double);
};

inline LennardJones::LennardJones(double eps, double sigma, double rCut):
epsx4(eps*4), epsx24(eps*24), sigma2(sigma*sigma),
rCut(rCut), rCut2(rCut * rCut)
{
	cutoffShift = _VnoCut(0, 0, rCut);
}

__host__ __device__ inline double LennardJones::_VnoCut(const double dx, const double dy, const double dz)
{
	const double r2 = dx*dx + dy*dy + dz*dz;
	
	double relR2 = sigma2 / r2;
	double relR6  = relR2 * relR2 * relR2;
	double relR12 = relR6 * relR6;
	return epsx4 * (relR12 - relR6);
}

__host__ __device__ inline double LennardJones::V(const double dx, const double dy, const double dz)
{
	const double r2 = dx*dx + dy*dy + dz*dz;
	
	if (r2 > rCut2) return 0;
	
	double relR2 = sigma2 / r2;
	double relR6  = relR2 * relR2 * relR2;
	double relR12 = relR6 * relR6;
	return epsx4 * (relR12 - relR6) - cutoffShift;
}

__host__ __device__ inline void LennardJones::F(const double dx, const double dy, const double dz, 
												double& fx,      double& fy,      double& fz)
{
	const double r2 = dx*dx + dy*dy + dz*dz;
	if (r2 > rCut2)
	{
		fx = 0;
		fy = 0;
		fz = 0;
		return;
	}
	
	double r_2 = 1.0 / r2;	
	double relR2  = sigma2 * r_2;
	double relR6  = relR2 * relR2 * relR2;
	double relR12 = relR6 * relR6;
	
	double fAbs = epsx24 * r_2 * (relR6 - 2*relR12);
	fx = fAbs * dx;
	fy = fAbs * dy;	
	fz = fAbs * dz;	
}


//**********************************************************************************************************************
// DPD
//**********************************************************************************************************************
class DPD
{
private:
	double alpha, gamma, temp, sigma, dt_1;
    double rCut2;
    
    mt19937 gen;
    normal_distribution<double> norm;
    
    inline double w(double);
    
public:
	double rCut;
	inline DPD(double, double, double, double, double, int);
    __host__ __device__ inline void   F(const double, const double, const double, const double, const double, const double, double&, double&, double&);
	__host__ __device__ inline double V(const double, const double, const double);
    //inline double _VnoCut(const double, const double, const double);
};

inline DPD::DPD(double alpha, double gamma, double temp, double rCut, double dt, int seed = 0):
alpha(alpha), gamma(gamma), temp(temp), rCut(rCut), rCut2(rCut * rCut),
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp)), dt_1(1.0/sqrt(dt))
{
}

inline double DPD::w(double r)
{
    //TODO implement with masking operations
    return (r<rCut) ? 1 - r/rCut : 0;
}

__host__ __device__ inline void DPD::F(const double dx, const double dy, const double dz,
                   const double vx, const double vy, const double vz,
                   double& fx,      double& fy,      double& fz)
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
    double fd = gamma * wr * wr * (dx*vx + dy*vy + dz*vz) / r2;   // !!! minus
    double fr = sigma * wr * norm(gen) * IrI_1 * dt_1;
    
    double fAbs = fc + fd + fr;
    fx = fAbs * dx;
	fy = fAbs * dy;
	fz = fAbs * dz;
}

__host__ __device__ inline double DPD::V(const double dx, const double dy, const double dz)
{
    return 0;
}



//**********************************************************************************************************************
// SEM+DPD
//**********************************************************************************************************************
class SEM
{
private:
	double alpha, gamma, temp, sigma, dt_1;
    double u0, rho, req;
    double rCut2;
    
    mt19937 gen;
    normal_distribution<double> norm;
    
    inline double w(double);
    
public:
	double rCut;
	inline DPD(double, double, double, double, double, int);
    __host__ __device__ inline void   F(const double, const double, const double, const double, const double, const double, double&, double&, double&);
	__host__ __device__ inline double V(const double, const double, const double);
    //inline double _VnoCut(const double, const double, const double);
};

inline DPD::DPD(double alpha, double gamma, double temp, double rCut, double dt, int seed = 0):
alpha(alpha), gamma(gamma), temp(temp), rCut(rCut), rCut2(rCut * rCut),
gen(seed), norm(0, 1), sigma(sqrt(2 * gamma * temp)), dt_1(1.0/sqrt(dt))
{
}

inline double DPD::w(double r)
{
    //TODO implement with masking operations
    return (r<rCut) ? 1 - r/rCut : 0;
}

__host__ __device__ inline void DPD::F(const double dx, const double dy, const double dz,
                                       const double vx, const double vy, const double vz,
                                       double& fx,      double& fy,      double& fz)
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
    double fd = gamma * wr * wr * (dx*vx + dy*vy + dz*vz) / r2;   // !!! minus
    double fr = sigma * wr * norm(gen) * IrI_1 * dt_1;
    
    double fAbs = fc + fd + fr;
    fx = fAbs * dx;
	fy = fAbs * dy;
	fz = fAbs * dz;
}

__host__ __device__ inline double DPD::V(const double dx, const double dy, const double dz)
{
    return 0;
}






