/*
 *  Cpukernels.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "CellList.h"
#include "Particles.h"
#include "Interaction.h"

#include "Misc.h"

#include "dpd/cuda-dpd.h"
#include "sem/cuda-sem.h"

using namespace std;

inline void _allocate(Particles* part)
{
	int n = part->n;
	part->xdata  = new real[n];
	part->ydata  = new real[n];
	part->zdata  = new real[n];
	part->vxdata  = new real[n];
	part->vydata  = new real[n];
	part->vzdata  = new real[n];
    part->axdata  = new real[n];
	part->aydata  = new real[n];
	part->azdata  = new real[n];
    
	part->m  = new real[n];
    
    part->label = new int[n];
}

template<typename T>
inline void _fill(T* x, int n, T val)
{
	for (int i=0; i<n; i++)
		x[i] = val;
}

inline void _scal(real *y, int n, real factor)
{
	for (int i=0; i<n; i++)
		y[i] *= factor;
}

inline void _axpy(real *y, real *x, int n, real a)
{
	for (int i=0; i<n; i++)
		y[i] += a * x[i];
}

inline void _nuscal(real *y, real *x, int n) // y[i] = y[i] / x[i]
{
	for (int i=0; i<n; i++)
		y[i] /= x[i/3];
}

inline void _normalize(real *x, int n, real xlo, real xhi, int sh=0)
{
    real len = xhi - xlo;
	for (int i=0; i<n; i++)
	{
        int ind = 3*i + sh;
		if (x[ind] > xhi) x[ind] -= len;
		if (x[ind] < xlo) x[ind] += len;
	}
}

inline real _kineticNrg(Particles* part)
{
	real res = 0;
	
	for (int i=0; i<part->n; i++)
		res += part->m[i] * (part->vx(i)*part->vx(i) +
							 part->vy(i)*part->vy(i) +
							 part->vz(i)*part->vz(i));
	return res*0.5;
}

inline void _linMomentum(Particles* part, real& px, real& py, real& pz)
{
	px = py = pz = 0;
	
	for (int i=0; i<part->n; i++)
	{
		px += part->m[i] * part->vx(i);
		py += part->m[i] * part->vy(i);
		pz += part->m[i] * part->vz(i);
	}
}

inline void cross(real  ax, real  ay, real  az,
				  real  bx, real  by, real  bz,
				  real& rx, real& ry, real& rz)
{
	rx = ay*bz - az*by;
	ry = az*bx - ax*bz;
	rz = ax*by - ay*bx;
}

inline void _angMomentum(Particles* part, real& Lx, real& Ly, real& Lz)
{
	Lx = Ly = Lz = 0;
	real lx, ly, lz;
	
	for (int i=0; i<part->n; i++)
	{
		cross(part->x(i),  part->y(i),  part->z(i),
			  part->vx(i), part->vy(i), part->vz(i),
			  lx,          ly,          lz);
		Lx += part->m[i] * lx;
		Ly += part->m[i] * ly;
		Lz += part->m[i] * lz;
	}
}


inline void _centerOfMass(Particles* part, real& mx, real& my, real& mz)
{
	real totM = 0;
	mx = my = mz = 0;
	
	for (int i=0; i<part->n; i++)
	{
		mx += part->m[i] * part->x(i);
		my += part->m[i] * part->y(i);
		mz += part->m[i] * part->z(i);
		
		totM += part->m[i];
	}
	
	mx /= totM;
	my /= totM;
	mz /= totM;
}

template<typename Interactor>
void _computeForces(Cells<Particles>* cells,
                    real* x,  real* y,  real *z,
                    real* vx, real* vy, real* vz,
                    real* ax, real* ay, real* az,
                    int n, Interactor inter)
{
}

template<>
void _computeForces<DPD>(Cells<Particles>* cells,
                         real* x,  real* y,  real* z,
                         real* vx, real* vy, real* vz,
                         real* ax, real* ay, real* az,
                         int n, DPD dpd)
{
    forces_dpd_cuda(x, y, z,  vx, vy, vz,  ax, ay, az,  n,  dpd.rc,
                    cells->xHigh0 - cells->xLow0, cells->xHigh1 - cells->xLow1, cells->xHigh2 - cells->xLow2,
                    dpd.alpha, dpd.gamma, dpd.sigma, 1.0);
}

template<>
void _computeForces<SEM>(Cells<Particles>* cells,
                         real* x,  real* y,  real* z,
                         real* vx, real* vy, real* vz,
                         real* ax, real* ay, real* az,
                         int n, SEM sem)
{
    real dt = 2*sem.gamma*sem.temp / sem.sigma / sem.sigma;
    
    forces_sem_cuda(x, y, z,  vx, vy, vz,  ax, ay, az,  n,  sem.rCut,
                    cells->xHigh0 - cells->xLow0, cells->xHigh1 - cells->xLow1, cells->xHigh2 - cells->xLow2,
                    sem.gamma, sem.temp, dt, sem.u0, sem.rho, 1.0/sem.req_1, sem.D, sem.rc);
}


template<typename Interactor>
void _computeForces (Cells<Particles>* cells,
                     real* x1,  real* y1,  real* z1,
                     real* vx1, real* vy1, real* vz1,
                     real* ax1, real* ay1, real* az1,
                     int n1,
                     real* x2,  real* y2,  real* z2,
                     real* vx2, real* vy2, real* vz2,
                     real* ax2, real* ay2, real* az2,
                     int n2,
                     Interactor inter)
{
}


template<>
void _computeForces<DPD>(Cells<Particles>* cells,
                         real* x1,  real* y1,  real* z1,
                         real* vx1, real* vy1, real* vz1,
                         real* ax1, real* ay1, real* az1,
                         int n1,
                         real* x2,  real* y2,  real* z2,
                         real* vx2, real* vy2, real* vz2,
                         real* ax2, real* ay2, real* az2,
                         int n2,
                         DPD dpd)
{
    forces_dpd_cuda_bipartite(x1, y1, z1,  vx1, vy1, vz1,  ax1, ay1, az1,  n1, 0,
                              x2, y2, z2,  vx2, vy2, vz2,  ax2, ay2, az2,  n2, n1,
                              dpd.rc,
                              cells->xHigh0 - cells->xLow0, cells->xHigh1 - cells->xLow1, cells->xHigh2 - cells->xLow2,
                              dpd.alpha, dpd.gamma, dpd.sigma, 1);
}


template<typename Interactor>
void computeForces(Particles** part, Cells<Particles>*** cells, int a, int b, Interactor &inter, int time)
{
    if (a == b)
    {
        _computeForces<Interactor>(cells[a][a],
                                   part[a]->xdata,  part[a]->ydata,  part[a]->zdata,
                                   part[a]->vxdata, part[a]->vydata, part[a]->vzdata,
                                   part[a]->axdata, part[a]->aydata, part[a]->azdata, part[a]->n, inter);
    }
    else
    {
        _computeForces<Interactor>(cells[a][b],
                                   part[a]->xdata,  part[a]->ydata,  part[a]->zdata,
                                   part[a]->vxdata, part[a]->vydata, part[a]->vzdata,
                                   part[a]->axdata, part[a]->aydata, part[a]->azdata, part[a]->n,
                                   part[b]->xdata,  part[b]->ydata,  part[b]->zdata,
                                   part[b]->vxdata, part[b]->vydata, part[b]->vzdata,
                                   part[b]->axdata, part[b]->aydata, part[b]->azdata, part[b]->n, inter);
    }

}


inline void _rdf(Particles** part, Cells<Particles>*** cells, int a, real* bins, real h, int nBins, real vol)
{
    real dens = part[a]->n / vol;
    for (int i = 0; i<nBins; i++) bins[i] = 0;
    
    for (int ix=0; ix<cells[a][a]->n0; ix++)
        for (int iy=0; iy<cells[a][a]->n1; iy++)
            for (int iz=0; iz<cells[a][a]->n2; iz++)
            {
                int ij[3];
                real xAdd[3];
                
                int origIJ[3] = {ix, iy, iz};
                int srcId = cells[a][a]->getCellIndByIJ(origIJ);
                
                int srcBegin = cells[a][a]->pstart[srcId];
                int srcEnd   = cells[a][a]->pstart[srcId+1];
                
                // All but self-self
                for (int neigh = 0; neigh < 27; neigh++)
                {
                    if (neigh == 13) continue;
                    const int sh[3] = {neigh / 9 - 1, (neigh / 3) % 3 - 1, neigh % 3 - 1};
                    // Resolve periodicity
                    for (int k=0; k<3; k++)
                        ij[k] = origIJ[k] + sh[k];
                    
                    cells[a][a]->correct(ij, xAdd);
                    
                    int dstId    = cells[a][a]->getCellIndByIJ(ij);
                    int dstBegin = cells[a][a]->pstart[dstId];
                    int dstEnd   = cells[a][a]->pstart[dstId+1];
                    
                    for (int j=srcBegin; j<srcEnd; j++)
                    {
                        int src = cells[a][a]->pobjids[j];
                        
                        for (int k=dstBegin; k<dstEnd; k++)
                        {
                            int dst = cells[a][a]->pobjids[k];
                            
                            const real dx = part[a]->x(dst) + xAdd[0] - part[a]->x(src);
                            const real dy = part[a]->y(dst) + xAdd[1] - part[a]->y(src);
                            const real dz = part[a]->z(dst) + xAdd[2] - part[a]->z(src);
                            
                            const real r = sqrt(dx*dx + dy*dy + dz*dz);
                            int bin = (int) (r / h);
                            if (bin < nBins) bins[bin]++;
                        }
                    }
                }
                
                // And now self
                for (int j=srcBegin; j<srcEnd; j++)
                {
                    int src = cells[a][a]->pobjids[j];
                    for (int k=j+1; k<srcEnd; k++)
                    {
                        int dst = cells[a][a]->pobjids[k];
                        
                        const real dx = part[a]->x(dst) - part[a]->x(src);
                        const real dy = part[a]->y(dst) - part[a]->y(src);
                        const real dz = part[a]->z(dst) - part[a]->z(src);
                        
                        const real r = sqrt(dx*dx + dy*dy + dz*dz);
                        int bin = (int) (r / h);
                        if (bin < nBins) bins[bin]++;
                    }
                }
                
            }
    
    
    for (int i = 0; i<nBins; i++)
    {
        bins[i] /= part[a]->n * 4/3.0*M_PI * h*h*h * ( (i+1)*(i+1)*(i+1) - i*i*i ) * dens;
    }
}



