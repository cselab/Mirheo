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

#include "Misc.h"

inline void _allocate(Particles* part)
{
	int n = part->n;
	part->xdata  = new real[3*n];
	part->vdata  = new real[3*n];
	part->adata  = new real[3*n];
	   
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

/*inline real _periodize(real val, real l, real l_2)
{
  	const real vlow  = val - l_2;	
	val = vlow  - copysign(l_2, vlow);  // val > l_2  ? val - l : val
	
	const real vhigh = val + l_2;
	val = vhigh - copysign(l_2, vhigh); // val < -l_2 ? val + l : val

	return val;
}*/


inline real _periodize(real val, real lo, real hi, real size)
{
	if (val > hi)
		return val - size;
	if (val < lo)
		return val + size;
	return val;
}

namespace Forces
{
    struct Arguments
    {
        static Particles** part;
        static real xlo, xhi, ylo, yhi, zlo, zhi;
        static Cells<Particles>** cells;
        static vector<real> rCuts;
    };
    
    template<int a, int b>
    struct _N2 : Arguments
    {
        void operator()()
        {
            real fx, fy, fz;
            real dx, dy, dz;
            real vx, vy, vz;
            
            if (part[a]->n <= 0 || part[b]->n <= 0) return;
            
            real sizex = xhi - xlo;
            real sizey = yhi - yhi;
            real sizez = zhi - zlo;
            
            for (int i = 0; i<part[a]->n; i++)
            {
                //UnrollerP<>::step( ((a == b) ? i+1 : 0), part[b]->n, [&] (int j)
                for (int j = ((a == b) ? i+1 : 0); j<part[b]->n; j++)
                {
                    dx = _periodize(part[b]->x(j) - part[a]->x(i), xlo, xhi, sizex);
                    dy = _periodize(part[b]->y(j) - part[a]->y(i), ylo, yhi, sizey);
                    dz = _periodize(part[b]->z(j) - part[a]->z(i), zlo, zhi, sizez);
                    
                    vx = part[b]->vx(j) - part[a]->vx(i);
                    vy = part[b]->vy(j) - part[a]->vy(i);
                    vz = part[b]->vz(j) - part[a]->vz(i);
                    
                    force<a, b>(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
                    
                    part[a]->ax(i) += fx;
                    part[a]->ay(i) += fy;
                    part[a]->az(i) += fz;
                    
                    part[b]->ax(j) -= fx;
                    part[b]->ay(j) -= fy;
                    part[b]->az(j) -= fz;
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
            real xAdd[3];

            real fx, fy, fz;
            
            if (part[a]->n <= 0 || part[b]->n <= 0) return;

            // Loop over all the particles
            for (int i=0; i<part[a]->n; i++)
            {
                real x = part[a]->x(i);
                real y = part[a]->y(i);
                real z = part[a]->z(i);

                // Get id of the cell and its coordinates
                int cid = cells[b]->which(x, y, z);
                cells[b]->getCellIJByInd(cid, origIJ);

                // Loop over all the 27 neighboring cells
                for (sh[0]=-1; sh[0]<=1; sh[0]++)
                    for (sh[1]=-1; sh[1]<=1; sh[1]++)
                        for (sh[2]=-1; sh[2]<=1; sh[2]++)
                        {
                            // Resolve periodicity
                            for (int k=0; k<3; k++)
                                ij[k] = origIJ[k] + sh[k];

                            cells[b]->correct(ij, xAdd);

                            cid = cells[b]->getCellIndByIJ(ij);
                            int begin = cells[b]->pstart[cid];
                            int end   = cells[b]->pstart[cid+1];

                            //Unroll::UnrollerP<>::step( begin, end, [&] (int j)
                            for (int j=begin; j<end; j++)
                            {
                                int neigh = cells[b]->pobjids[j];
                                if (a != b || i > neigh)
                                {
                                    debug("%d %d\n", i, neigh);

                                    real dx = part[b]->x(neigh) + xAdd[0] - x;
                                    real dy = part[b]->y(neigh) + xAdd[1] - y;
                                    real dz = part[b]->z(neigh) + xAdd[2] - z;
                                    
                                    real vx = part[b]->vx(neigh) - part[a]->vx(i);
                                    real vy = part[b]->vy(neigh) - part[a]->vy(i);
                                    real vz = part[b]->vz(neigh) - part[a]->vz(i);
                                    
                                    force<a, b>(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
                                    
                                    part[a]->ax(i) += fx;
                                    part[a]->ay(i) += fy;
                                    part[a]->az(i) += fz;
                                    
                                    part[b]->ax(neigh) -= fx;
                                    part[b]->ay(neigh) -= fy;
                                    part[b]->az(neigh) -= fz;
                                }
                            }
                        }
            }
        }
    };
    
    
    template<int a, int b>
    struct _Cells1 : Arguments
    {
        static const int stride = 2;
        void exec(int sx, int sy, int sz)
        {
            real rCut2 = rCuts[a] * rCuts[a];
            
            if (part[a]->n <= 0 || part[b]->n <= 0) return;
            
//#pragma omp parallel for collapse(3)
            for (int ix=sx; ix<cells[a]->n0; ix+=stride)
                for (int iy=sy; iy<cells[a]->n1; iy+=stride)
                    for (int iz=sz; iz<cells[a]->n2; iz+=stride)
#pragma omp task
                    {
                        Cells<Particles>& c = *cells[a];

                        int ij[3];
                        real xAdd[3];
                        real fx, fy, fz;
                        
                        int origIJ[3] = {ix, iy, iz};
                        int srcId = c.getCellIndByIJ(origIJ);
                        
                        int srcBegin = c.pstart[srcId];
                        int srcEnd   = c.pstart[srcId+1];
                        
                        // All but self-self
                        for (int neigh = 0; neigh < 13; neigh++)
                        {
                            const int sh[3] = {neigh / 9 - 1, (neigh / 3) % 3 - 1, neigh % 3 - 1};
                            
                            // Resolve periodicity
                            for (int k=0; k<3; k++)
                                ij[k] = origIJ[k] + sh[k];
                            
                            c.correct(ij, xAdd);
                            
                            int dstId    = c.getCellIndByIJ(ij);
                            int dstBegin = c.pstart[dstId];
                            int dstEnd   = c.pstart[dstId+1];
                            
                            for (int j=srcBegin; j<srcEnd; j++)
                            {
                                int src = c.pobjids[j];
                                for (int k=dstBegin; k<dstEnd; k++)
                                {
                                    int dst = c.pobjids[k];

                                    const real dx = part[a]->x(dst) + xAdd[0] - part[a]->x(src);
                                    const real dy = part[a]->y(dst) + xAdd[1] - part[a]->y(src);
                                    const real dz = part[a]->z(dst) + xAdd[2] - part[a]->z(src);
                                    
                                    const real r2 = dx*dx + dy*dy + dz*dz;
                                    if (r2 < rCut2)
                                    {
                                        real vx = part[a]->vx(dst) - part[a]->vx(src);
                                        real vy = part[a]->vy(dst) - part[a]->vy(src);
                                        real vz = part[a]->vz(dst) - part[a]->vz(src);
                                        
                                        force<a, b>(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);

                                        part[a]->ax(src) += fx;
                                        part[a]->ay(src) += fy;
                                        part[a]->az(src) += fz;
                                        
                                        part[a]->ax(dst) -= fx;
                                        part[a]->ay(dst) -= fy;
                                        part[a]->az(dst) -= fz;
                                    }
                                }
                            }
                        }
                        
                        // And now self-self
                        for (int j=srcBegin; j<srcEnd; j++)
                        {
                            int src = c.pobjids[j];
                            for (int k=j+1; k<srcEnd; k++)
                            {
                                int dst = c.pobjids[k];
                                debug("%d %d\n", src, dst);

                                
                                const real dx = part[a]->x(dst) - part[a]->x(src);
                                const real dy = part[a]->y(dst) - part[a]->y(src);
                                const real dz = part[a]->z(dst) - part[a]->z(src);
                                
                                const real r2 = dx*dx + dy*dy + dz*dz;
                                if (r2 < rCut2)
                                {
                                    real vx = part[a]->vx(dst) - part[a]->vx(src);
                                    real vy = part[a]->vy(dst) - part[a]->vy(src);
                                    real vz = part[a]->vz(dst) - part[a]->vz(src);
                                    
                                    force<a, b>(dx, dy, dz,  vx, vy, vz,  fx, fy, fz);
                                    
                                    part[a]->ax(src) += fx;
                                    part[a]->ay(src) += fy;
                                    part[a]->az(src) += fz;
                                    
                                    part[a]->ax(dst) -= fx;
                                    part[a]->ay(dst) -= fy;
                                    part[a]->az(dst) -= fz;
                                }
                            }
                        }
                    }
            #pragma omp taskwait
        }
        
        void operator()()
        {
            for (int i=0; i<stride; i++)
                for (int j=0; j<stride; j++)
                    for (int k=0; k<stride; k++)
                        exec(i, j, k);
        }
    };



}


void _normalize(real *x, int n, real xlo, real xhi, int sh=0)
{
    real len = xhi - xlo;
	for (int i=0; i<n; i++)
	{
        int ind = 3*i + sh;
		if (x[ind] > xhi) x[ind] -= len;
		if (x[ind] < xlo) x[ind] += len;
	}
}

real _kineticNrg(Particles* part)
{
	real res = 0;	
	
	for (int i=0; i<part->n; i++)
		res += part->m[i] * (part->vx(i)*part->vx(i) +
							 part->vy(i)*part->vy(i) +
							 part->vz(i)*part->vz(i));
	return res*0.5;
}

void _linMomentum(Particles* part, real& px, real& py, real& pz)
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

void _angMomentum(Particles* part, real& Lx, real& Ly, real& Lz)
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


void _centerOfMass(Particles* part, real& mx, real& my, real& mz)
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




