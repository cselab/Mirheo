/*
 *  Cppkernels.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once


#include "CellList.h"
#include "Potential.h"

inline void _allocate(Particles* part)
{
	int n = part->n;
	part->x  = new double[n];
	part->y  = new double[n];
	part->z  = new double[n];
	part->vx = new double[n];
	part->vy = new double[n];
	part->vz = new double[n];
	part->ax = new double[n];
	part->ay = new double[n];
	part->az = new double[n];
	part->m  = new double[n];
}

inline void _fill(double* x, int n, double val)
{
	for (int i=0; i<n; i++)
		x[i] = val;
}

inline void _scal(double *y, int n, double factor)
{
	for (int i=0; i<n; i++)
		y[i] *= factor;
}

inline void _K1(double *y, double *x, int n, double a) // BLAS lev 1: axpy
{
	for (int i=0; i<n; i++)
		y[i] += a * x[i];
}

/*inline double _periodize(double val, double l, double l_2)
{
  	const double vlow  = val - l_2;	
	val = vlow  - copysign(l_2, vlow);  // val > l_2  ? val - l : val
	
	const double vhigh = val + l_2;
	val = vhigh - copysign(l_2, vhigh); // val < -l_2 ? val + l : val

	return val;
}*/


inline double _periodize(double val, double l, double l_2)
{
	if (val > l_2)
		return val - l;
	if (val < -l_2)
		return val + l;
	return val;
}

void _K2(Particles* part, DPD* potential, double L)
{
	_fill(part->ax, part->n, 0);
	_fill(part->ay, part->n, 0);
	_fill(part->az, part->n, 0);
	
	double fx, fy, fz;
	double dx, dy, dz;
    double vx, vy, vz;
	double L_2 = 0.5*L;

	for (int i=0; i<part->n; i++)
	{
		for (int j = i+1; j<part->n; j++)
		{
			dx = _periodize(part->x[j] - part->x[i], L, L_2);
			dy = _periodize(part->y[j] - part->y[i], L, L_2);
			dz = _periodize(part->z[j] - part->z[i], L, L_2);
            
            vx = part->vx[j] - part->vx[i];
            vy = part->vy[j] - part->vy[i];
            vz = part->vz[j] - part->vz[i];
			
			potential->F(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
			
			part->ax[i] += fx;
			part->ay[i] += fy;
			part->az[i] += fz;
			
			part->ax[j] -= fx;
			part->ay[j] -= fy;
			part->az[j] -= fz;
		}
		
		const double m_1 = 1.0 / part->m[i];
		part->ax[i] *= m_1;
		part->ay[i] *= m_1;
		part->az[i] *= m_1;
	}
}

void _K2(Particles* part, DPD* potential, Cells<Particles>* cells, double L)
{
	int origIJ[3];
	int ij[3], sh[3];
	double xAdd[3];
	
	_fill(part->ax, part->n, 0);
	_fill(part->ay, part->n, 0);
	_fill(part->az, part->n, 0);
	
	double fx, fy, fz;
	
	// Loop over all the particles
	for (int i=0; i<part->n; i++)
	{
		double x = part->x[i];
		double y = part->y[i];
		double z = part->z[i];

		// Get id of the cell and its coordinates
		int cid = cells->which(x, y, z);
		cells->getCellIJByInd(cid, origIJ);
		
		// Loop over all the 27 neighboring cells
		for (sh[0]=-1; sh[0]<=1; sh[0]++)
			for (sh[1]=-1; sh[1]<=1; sh[1]++)
				for (sh[2]=-1; sh[2]<=1; sh[2]++)
				{
					// Resolve periodicity
					for (int k=0; k<3; k++)
						ij[k] = origIJ[k] + sh[k];

					cells->correct(ij, xAdd);
						
					cid = cells->getCellIndByIJ(ij);
					int begin = cells->pstart[cid];
					int end   = cells->pstart[cid+1];
					
					for (int j=begin; j<end; j++)
					{
						int neigh = cells->pobjids[j];
						if (i > neigh)
						{
							double dx = part->x[neigh] + xAdd[0] - x;
							double dy = part->y[neigh] + xAdd[1] - y;
							double dz = part->z[neigh] + xAdd[2] - z;
							
                            double vx = part->vx[neigh] - part->vx[i];
                            double vy = part->vy[neigh] - part->vy[i];
                            double vz = part->vz[neigh] - part->vz[i];
                            
                            potential->F(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
							
							part->ax[i] += fx;
							part->ay[i] += fy;
							part->az[i] += fz;
                            
                            part->ax[neigh] -= fx;
							part->ay[neigh] -= fy;
							part->az[neigh] -= fz;
						}
					}
				}
		
		const double m_1 = 1.0 / part->m[i];
		part->ax[i] *= m_1;
		part->ay[i] *= m_1;
		part->az[i] *= m_1;
	}
}

void _normalize(double *x, double n, double x0, double xmax)
{
	for (int i=0; i<n; i++)
	{
		if (x[i] > xmax) x[i] -= xmax - x0;
		if (x[i] < x0)   x[i] += xmax - x0;
	}
}

double _kineticNrg(Particles* part)
{
	double res = 0;	
	
	for (int i=0; i<part->n; i++)
		res += part->m[i] * (part->vx[i]*part->vx[i] +
							 part->vy[i]*part->vy[i] +
							 part->vz[i]*part->vz[i]);
	return res*0.5;
}

double _potentialNrg(Particles* part, DPD* potential, double L)
{
	double res = 0;
	double L_2 = 0.5*L;
	double dx, dy, dz;
	
	for (int i=0; i<part->n; i++)
	{				
		for (int j=i+1; j<part->n; j++)
		{
			dx = _periodize(part->x[j] - part->x[i], L, L_2);
			dy = _periodize(part->y[j] - part->y[i], L, L_2);
			dz = _periodize(part->z[j] - part->z[i], L, L_2);
			
			res += potential->V(dx, dy, dz);
		}
		
	}
	return res;
}

double _potentialNrg(Particles* part, DPD* potential, Cells<Particles>* cells, double L)
{
	int origIJ[3];
	int ij[3], sh[3];
	double xAdd[3];
	
	double res = 0;
	
	// Loop over all the particles
	for (int i=0; i<part->n; i++)
	{
		double x = part->x[i];
		double y = part->y[i];
		double z = part->z[i];

		// Get id of the cell and its coordinates
		int cid = cells->which(x, y, z);
		cells->getCellIJByInd(cid, origIJ);
		
		// Loop over all the 27 neighboring cells
		for (sh[0]=-1; sh[0]<=1; sh[0]++)
			for (sh[1]=-1; sh[1]<=1; sh[1]++)
				for (sh[2]=-1; sh[2]<=1; sh[2]++)
				{
					// Resolve periodicity
					for (int k=0; k<3; k++)
						ij[k] = origIJ[k] + sh[k];

					cells->correct(ij, xAdd);
					
					cid = cells->getCellIndByIJ(ij);
					int begin = cells->start[cid];
					int end   = cells->start[cid+1];
					
					for (int j=begin; j<end; j++)
					{
						int neigh = cells->objids[j];
						if (i != neigh)
						{
							double dx = part->x[neigh] + xAdd[0] - x;
							double dy = part->y[neigh] + xAdd[1] - y;
							double dz = part->z[neigh] + xAdd[2] - z;
							
							res += potential->V(dx, dy, dz);
						}
					}
				}
	}
	
	return res*0.5;
}

void _linMomentum(Particles* part, double& px, double& py, double& pz)
{
	px = py = pz = 0;
	
	for (int i=0; i<part->n; i++)
	{
		px += part->m[i] * part->vx[i];
		py += part->m[i] * part->vy[i];
		pz += part->m[i] * part->vz[i];
	}
}

inline void cross(double  ax, double  ay, double  az,
				  double  bx, double  by, double  bz,
				  double& rx, double& ry, double& rz)
{
	rx = ay*bz - az*by;
	ry = az*bx - ax*bz;
	rz = ax*by - ay*bx;
}

void _angMomentum(Particles* part, double& Lx, double& Ly, double& Lz)
{
	Lx = Ly = Lz = 0;
	double lx, ly, lz;
	
	for (int i=0; i<part->n; i++)
	{
		cross(part->x[i],  part->y[i],  part->z[i],
			  part->vx[i], part->vy[i], part->vz[i], 
			  lx,          ly,          lz);
		Lx += part->m[i] * lx;
		Ly += part->m[i] * ly;
		Lz += part->m[i] * lz;
	}
}


void _centerOfMass(Particles* part, double& mx, double& my, double& mz)
{
	double totM = 0;
	mx = my = mz = 0;
	
	for (int i=0; i<part->n; i++)
	{
		mx += part->m[i] * part->x[i];
		my += part->m[i] * part->y[i];
		mz += part->m[i] * part->z[i];
		
		totM += part->m[i];
	}
	
	mx /= totM;
	my /= totM;
	mz /= totM;
}




