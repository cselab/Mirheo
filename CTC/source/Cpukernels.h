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
#include "Particles.h"

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

inline void _axpy(double *y, double *x, int n, double a)
{
	for (int i=0; i<n; i++)
		y[i] += a * x[i];
}

inline void _nuscal(double *y, double *x, int n) // y[i] = y[i] / x[i]
{
	for (int i=0; i<n; i++)
		y[i] /= x[i];
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

namespace Forces
{
    struct Arguments
    {
        static Particles** part;
        static double L;
        static Cells<Particles>** cells;
    };
    
    template<int a, int b>
    struct _N2 : Arguments
    {
        void operator()()
        {
            double fx, fy, fz;
            double dx, dy, dz;
            double vx, vy, vz;
            double L_2 = 0.5*L;
            
            for (int i = 0; i<part[a]->n; i++)
            {
                //UnrollerP<>::step( ((a == b) ? i+1 : 0), part[b]->n, [&] (int j)
                for (int j = ((a == b) ? i+1 : 0); j<part[b]->n; j++)
                {
                    dx = _periodize(part[b]->x[j] - part[a]->x[i], L, L_2);
                    dy = _periodize(part[b]->y[j] - part[a]->y[i], L, L_2);
                    dz = _periodize(part[b]->z[j] - part[a]->z[i], L, L_2);
                    
                    vx = part[b]->vx[j] - part[a]->vx[i];
                    vy = part[b]->vy[j] - part[a]->vy[i];
                    vz = part[b]->vz[j] - part[a]->vz[i];
                    
                    force<a, b>(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
                    
                    part[a]->ax[i] += fx;
                    part[a]->ay[i] += fy;
                    part[a]->az[i] += fz;
                    
                    part[b]->ax[j] -= fx;
                    part[b]->ay[j] -= fy;
                    part[b]->az[j] -= fz;
                }
            }
        }

    };
    
    
    template<int a, int b>
    struct _Cells : Arguments
    {
        void operator()()
        {
            int origIJ[3];
            int ij[3], sh[3];
            double xAdd[3];

            double fx, fy, fz;

            // Loop over all the particles
            for (int i=0; i<part[a]->n; i++)
            {
                double x = part[a]->x[i];
                double y = part[a]->y[i];
                double z = part[a]->z[i];

                // Get id of the cell and its coordinates
                int cid = cells[a]->which(x, y, z);
                cells[a]->getCellIJByInd(cid, origIJ);

                // Loop over all the 27 neighboring cells
                for (sh[0]=-1; sh[0]<=1; sh[0]++)
                    for (sh[1]=-1; sh[1]<=1; sh[1]++)
                        for (sh[2]=-1; sh[2]<=1; sh[2]++)
                        {
                            // Resolve periodicity
                            for (int k=0; k<3; k++)
                                ij[k] = origIJ[k] + sh[k];

                            cells[a]->correct(ij, xAdd);

                            cid = cells[a]->getCellIndByIJ(ij);
                            int begin = cells[b]->pstart[cid];
                            int end   = cells[b]->pstart[cid+1];

                            for (int j=begin; j<end; j++)
                            {
                                int neigh = cells[b]->pobjids[j];
                                if (a != b || i > neigh)
                                {
                                    double dx = part[b]->x[neigh] + xAdd[0] - x;
                                    double dy = part[b]->y[neigh] + xAdd[1] - y;
                                    double dz = part[b]->z[neigh] + xAdd[2] - z;

                                    double vx = part[b]->vx[neigh] - part[a]->vx[i];
                                    double vy = part[b]->vy[neigh] - part[a]->vy[i];
                                    double vz = part[b]->vz[neigh] - part[a]->vz[i];
                                    
                                    force<a, b>(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
                                    //if (a != b && fz > 0.001) printf("%f\n", fz);
                                    
                                    part[a]->ax[i] += fx;
                                    part[a]->ay[i] += fy;
                                    part[a]->az[i] += fz;
                                    
                                    part[b]->ax[neigh] -= fx;
                                    part[b]->ay[neigh] -= fy;
                                    part[b]->az[neigh] -= fz;
                                }
                            }
                        }
            }
        }
    };


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




