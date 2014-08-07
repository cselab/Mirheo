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

#include "Misc.h"

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

inline void _normalize(real *x, int n, real xlo, real xhi)
{
    real len = xhi - xlo;
	for (int i=0; i<n; i++)
	{
		if (x[i] > xhi) x[i] -= len;
		if (x[i] < xlo) x[i] += len;
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
inline void exec(Particles** part, Cells<Particles> &cellsA, Cells<Particles> &cellsB, int a, int b, Interactor &inter,
                 int sx, int sy, int sz, vector<vector<int>> pass, real* res, int t)
{
    static const int stride = 2;
    
    for (int ix=sx; ix<cellsA.n0; ix+=stride)
        for (int iy=sy; iy<cellsA.n1; iy+=stride)
            for (int iz=sz; iz<cellsA.n2; iz+=stride)
            {
                int ij[3];
                real xAdd[3];
                real fx, fy, fz;
                
                int origIJ[3] = {ix, iy, iz};
                int srcId = cellsA.getCellIndByIJ(origIJ);
                
                int srcBegin = cellsA.pstart[srcId];
                int srcEnd   = cellsA.pstart[srcId+1];
                
                // All but self-self
                for (int neigh = 0; neigh < pass.size(); neigh++)
                {
                    // Resolve periodicity
                    for (int k=0; k<3; k++)
                        ij[k] = origIJ[k] + pass[neigh][k];
                    
                    cellsA.correct(ij, xAdd);
                    
                    int dstId    = cellsB.getCellIndByIJ(ij);
                    int dstBegin = cellsB.pstart[dstId];
                    int dstEnd   = cellsB.pstart[dstId+1];
                    
                    for (int j=srcBegin; j<srcEnd; j++)
                    {
                        int src = cellsA.pobjids[j];
                        
                        for (int k=dstBegin; k<dstEnd; k++)
                        {
                            int dst = cellsB.pobjids[k];
                            
                            const real dx = part[b]->x(dst) + xAdd[0] - part[a]->x(src);
                            const real dy = part[b]->y(dst) + xAdd[1] - part[a]->y(src);
                            const real dz = part[b]->z(dst) + xAdd[2] - part[a]->z(src);
                            
                            const real r2 = dx*dx + dy*dy + dz*dz;
                            if (inter.nonzero(r2))
                            {
                                inter.F(dx, dy, dz,  part[a], part[b],  r2,  fx, fy, fz,  src, dst, t);
                                
                                part[a]->ax(src) += fx;
                                part[a]->ay(src) += fy;
                                part[a]->az(src) += fz;
                                
                                res[3*dst + 0] -= fx;
                                res[3*dst + 1] -= fy;
                                res[3*dst + 2] -= fz;
                                
                                //printf("%d  :  %d     %f %f %f        %f %f %f\n", t, src,  dx, dy, dz,  fx, fy, fz);
                            }
                        }
                    }
                }
            }
}

template<typename Interactor>
inline void exec(Particles** part, Cells<Particles> &cellsA, int a, Interactor &inter,
                 int sx, int sy, int sz, vector<vector<int>> pass, real* res, int t, bool myself)
{
    static const int stride = 2;
    
    if (a == 0) return;
    
    for (int ix=sx; ix<cellsA.n0; ix+=stride)
        for (int iy=sy; iy<cellsA.n1; iy+=stride)
            for (int iz=sz; iz<cellsA.n2; iz+=stride)
            {
                int ij[3];
                real xAdd[3];
                real fx, fy, fz;
                
                int origIJ[3] = {ix, iy, iz};
                int srcId = cellsA.getCellIndByIJ(origIJ);
                
                int srcBegin = cellsA.pstart[srcId];
                int srcEnd   = cellsA.pstart[srcId+1];
                
                // All but self-self
                for (int neigh = 0; neigh < pass.size(); neigh++)
                {
                    // Resolve periodicity
                    for (int k=0; k<3; k++)
                        ij[k] = origIJ[k] + pass[neigh][k];
                    
                    cellsA.correct(ij, xAdd);
                    
                    int dstId    = cellsA.getCellIndByIJ(ij);
                    int dstBegin = cellsA.pstart[dstId];
                    int dstEnd   = cellsA.pstart[dstId+1];
                    
                    for (int j=srcBegin; j<srcEnd; j++)
                    {
                        int src = cellsA.pobjids[j];
                        
                        for (int k=dstBegin; k<dstEnd; k++)
                        {
                            int dst = cellsA.pobjids[k];
                            
                            const real dx = part[a]->x(dst) + xAdd[0] - part[a]->x(src);
                            const real dy = part[a]->y(dst) + xAdd[1] - part[a]->y(src);
                            const real dz = part[a]->z(dst) + xAdd[2] - part[a]->z(src);
                            
                            const real r2 = dx*dx + dy*dy + dz*dz;
                            if (inter.nonzero(r2))
                            {
                                inter.F(dx, dy, dz,  part[a], part[a],  r2,  fx, fy, fz,  src, dst, t);
                                
                                res[3*src + 0] += fx;
                                res[3*src + 1] += fy;
                                res[3*src + 2] += fz;
                                
                                res[3*dst + 0] -= fx;
                                res[3*dst + 1] -= fy;
                                res[3*dst + 2] -= fz;
                            }
                        }
                    }
                }
                
                // And now self
                if (myself)
                    for (int j=srcBegin; j<srcEnd; j++)
                    {
                        int src = cellsA.pobjids[j];
                        for (int k=j+1; k<srcEnd; k++)
                        {
                            int dst = cellsA.pobjids[k];
                            
                            const real dx = part[a]->x(dst) - part[a]->x(src);
                            const real dy = part[a]->y(dst) - part[a]->y(src);
                            const real dz = part[a]->z(dst) - part[a]->z(src);
                            
                            const real r2 = dx*dx + dy*dy + dz*dz;
                            if (inter.nonzero(r2))
                            {
                                inter.F(dx, dy, dz,  part[a], part[a],  r2,  fx, fy, fz,  src, dst, t);
                                
                                res[3*src + 0] += fx;
                                res[3*src + 1] += fy;
                                res[3*src + 2] += fz;
                                
                                res[3*dst + 0] -= fx;
                                res[3*dst + 1] -= fy;
                                res[3*dst + 2] -= fz;
                            }
                        }
                    }

            }
}


template<typename Interactor>
void _cpuForces(Particles** part, Cells<Particles>*** cells, int a, int b, Interactor &inter, int time)
{
    static const int stride = 2;

    const static vector<vector<int>> pass1 = { {-1, -1, -1}, {-1, -1,  0}, {-1,  0, -1}, {-1,  0,  0}, { 0, -1, -1}, { 0, -1,  0}, { 0,  0, -1} };
    const static vector<vector<int>> pass2 = { {-1, -1,  1}, {-1,  0,  1}, {-1,  1, -1}, {-1,  1,  0}, {-1,  1,  1}, { 0, -1,  1}, { 0,  0,  0} };
    const static vector<vector<int>> pass3 = { { 0,  0,  1}, { 0,  1,  0}, { 0,  1,  1}, { 1,  0,  0}, { 1,  0,  1}, { 1,  1,  0}, { 1,  1,  1} };
    const static vector<vector<int>> pass4 = { { 0,  1, -1}, { 1, -1, -1}, { 1, -1,  0}, { 1, -1,  1}, { 1,  0, -1}, { 1,  1, -1} };
    real **buffer;
    
    if (part[a]->n <= 0 || part[b]->n <= 0 ) return;

    buffer = new real*[8];
    for (int i=0; i<8; i++)
        buffer[i] = new real[3*part[b]->n];
    
    for (int i=0; i<8; i++)
        for (int j=0; j<3*part[b]->n; j++)
            buffer[i][j] = 0;
    
#pragma omp parallel for collapse(3)
    for (int i=0; i<stride; i++)
        for (int j=0; j<stride; j++)
            for (int k=0; k<stride; k++)
            {
                exec(part, *cells[a][b], *cells[b][a], a, b, inter, i, j, k, pass1, buffer[4*i + 2*j + k], time);
            }
    
#pragma omp parallel for collapse(3)
    for (int i=0; i<stride; i++)
        for (int j=0; j<stride; j++)
            for (int k=0; k<stride; k++)
            {
                exec(part, *cells[a][b], *cells[b][a], a, b, inter, i, j, k, pass2, buffer[4*i + 2*j + k], time);
            }
    
#pragma omp parallel for collapse(3)
    for (int i=0; i<stride; i++)
        for (int j=0; j<stride; j++)
            for (int k=0; k<stride; k++)
            {
                exec(part, *cells[a][b], *cells[b][a], a, b, inter, i, j, k, pass3, buffer[4*i + 2*j + k], time);
            }
    
#pragma omp parallel for collapse(3)
    for (int i=0; i<stride; i++)
        for (int j=0; j<stride; j++)
            for (int k=0; k<stride; k++)
            {
                exec(part, *cells[a][b], *cells[b][a], a, b, inter, i, j, k, pass4, buffer[4*i + 2*j + k], time);
            }

    
    for (int j=0; j<3*part[b]->n; j++)
    {
        buffer[0][j] += buffer[1][j];
        buffer[2][j] += buffer[3][j];
        buffer[4][j] += buffer[5][j];
        buffer[6][j] += buffer[7][j];
        
        buffer[0][j] += buffer[2][j];
        buffer[4][j] += buffer[6][j];
        
        buffer[0][j]  = buffer[0][j] + buffer[4][j];
    }
    
    for (int j=0; j<part[b]->n; j++)
    {
        part[b]->ax(j) += buffer[0][3*j + 0];
        part[b]->ay(j) += buffer[0][3*j + 1];
        part[b]->az(j) += buffer[0][3*j + 2];
    }
    
    for (int i=0; i<8; i++)
        delete[] buffer[i];
    delete[] buffer;
}

template<typename Interactor>
void _cpuForces(Particles** part, Cells<Particles>*** cells, int a, Interactor &inter, int time)
{
    static const int stride = 2;

    const static vector<vector<int>> pass1 = { {-1, -1, -1}, {-1, -1,  0}, {-1,  0, -1}, {-1,  0,  0}, { 0, -1, -1}, { 0, -1,  0}, { 0,  0, -1} };
    const static vector<vector<int>> pass2 = { {-1, -1,  1}, {-1,  0,  1}, {-1,  1, -1}, {-1,  1,  0}, {-1,  1,  1}, { 0, -1,  1} };
    real **buffer;

    if (part[a]->n <= 0) return;

    buffer = new real*[8];
    for (int i=0; i<8; i++)
        buffer[i] = new real[3*part[a]->n];
    
    for (int i=0; i<8; i++)
        for (int j=0; j<3*part[a]->n; j++)
            buffer[i][j] = 0;
    
#pragma omp parallel for collapse(3)
    for (int i=0; i<stride; i++)
        for (int j=0; j<stride; j++)
            for (int k=0; k<stride; k++)
            {
                exec(part, *cells[a][a], a, inter, i, j, k, pass1, buffer[4*i + 2*j + k], time, false);
            }
    
#pragma omp parallel for collapse(3)
    for (int i=0; i<stride; i++)
        for (int j=0; j<stride; j++)
            for (int k=0; k<stride; k++)
            {
                exec(part, *cells[a][a], a, inter, i, j, k, pass2, buffer[4*i + 2*j + k], time, true);
            }
    
    for (int j=0; j<3*part[a]->n; j++)
    {
        buffer[0][j] += buffer[1][j];
        buffer[2][j] += buffer[3][j];
        buffer[4][j] += buffer[5][j];
        buffer[6][j] += buffer[7][j];
        
        buffer[0][j] += buffer[2][j];
        buffer[4][j] += buffer[6][j];
        
        buffer[0][j]  = buffer[0][j] + buffer[4][j];
    }
    
    for (int j=0; j<part[a]->n; j++)
    {
        part[a]->ax(j) += buffer[0][3*j + 0];
        part[a]->ay(j) += buffer[0][3*j + 1];
        part[a]->az(j) += buffer[0][3*j + 2];
    }
    
    for (int i=0; i<8; i++)
        delete[] buffer[i];
    delete[] buffer;
}

template<typename Interactor>
void computeForces(Particles** part, Cells<Particles>*** cells, int a, int b, Interactor &inter, int time)
{
    if (a == b) _cpuForces(part, cells, a, inter, time);
    else        _cpuForces(part, cells, a, b, inter, time);
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



