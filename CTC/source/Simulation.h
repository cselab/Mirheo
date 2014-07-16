/*
 *  Simulation.h
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <list>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <errno.h>
#include <string>

#include "Particles.h"
#include "Profiler.h"
#include "Potential.h"
#include "timer.h"
#include "Misc.h"
#ifdef MD_USE_CELLLIST
#include "CellList.h"
#endif

#ifdef __MD_USE_CUDA__
#include "Gpukernels.h"
#else
#include "Cpukernels.h"
#endif

using namespace std;

template <int N>
class Saver;

//**********************************************************************************************************************
// Let no one ignorant of geometry enter here
//**********************************************************************************************************************

// http://www.codeproject.com/Articles/75423/Loop-Unrolling-over-Template-Arguments
template<int Begin, int End, int Step = 1>
struct Unroller {
    template<typename Action>
    static void step(Action& action) {
        action(Begin);
        Unroller<Begin+Step, End, Step>::step(action);
    }
};

template<int End, int Step>
struct Unroller<End, End, Step> {
    template<typename Action>
    static void step(Action& action) {
    }
};
// Mind the gap


template<int ibeg, int iend, int jbeg, int jend>
struct Unroller2 {
    static void step() {
#ifdef MD_USE_CELLLIST
        Forces::_Cells<ibeg, jbeg> force;
#else
        Forces::_N2<ibeg, jbeg> force;
#endif
        force();
        Unroller2<ibeg, iend, jbeg+1, jend>::step();
    }
};

template<int ibeg, int iend, int jend>
struct Unroller2<ibeg, iend, jend, jend> {
    static void step() {
        Unroller2<ibeg+1, iend, ibeg+1, jend>::step();
    }
};

template<int iend, int jend>
struct Unroller2<iend, iend, jend, jend> {
    static void step() {
    }
};

//**********************************************************************************************************************
// Simulation
//**********************************************************************************************************************

Particles** Forces::Arguments::part;
real Forces::Arguments::L;
#ifdef MD_USE_CELLLIST
Cells<Particles>** Forces::Arguments::cells;
#endif

template<int N>
class Simulation
{
private:
	int    step;
	real x0, y0, z0, xmax, ymax, zmax;
	real dt;
	real L;
	
	Particles* part[N];
	list<Saver<N>*>	savers;
	
#ifdef MD_USE_CELLLIST
	Cells<Particles>* cells[N];
#endif
		
	void velocityVerlet();
	
public:
	Profiler profiler;

	Simulation (vector<int>, real, vector<real>, real, real);
	void setLattice(real *x, real *y, real *z, real lx, real ly, real lz, int n);
	
	real Ktot(int = -1);
	void linMomentum (real& px, real& py, real& pz, int = -1);
	void angMomentum (real& px, real& py, real& pz, int = -1);
	void centerOfMass(real& mx, real& my, real& mz, int = -1);
	
	inline int getIter() { return step; }
	inline Particles** getParticles() { return part; }
	
	void getPositions(int&, real*, real*);
	void registerSaver(Saver<N>*, int);
	void runOneStep();
};


//**********************************************************************************************************************
// Saver
//**********************************************************************************************************************

template<int N>
class Saver
{
protected:
	int period;
	ofstream* file;
	static string folder;
	Simulation<N>* my;
    
    bool opened;
	
public:
	Saver (ostream* cOut) : file((ofstream*)cOut) { opened = false; }
	Saver (string fname)
	{
        if (fname == "screen")
        {
            file = (ofstream*)&cout;
            opened = false;
            return;
        }
        
		file = new ofstream((this->folder+fname).c_str());
        opened = true;
	}
	Saver (){};
    ~Saver () { if (!opened) file->close(); }
	
	inline  void setEnsemble(Simulation<N>* e);
	inline  void setPeriod(int);
	inline  int  getPeriod();
	
	inline static bool makedir(string name)
	{
		folder = name;
		if (mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) return false;
		return true;
	}
	
	virtual void exec() = 0;
};

template<int N>
inline void Saver<N>::setPeriod(int p)
{
	period = p;
}

template<int N>
inline int Saver<N>::getPeriod()
{
	return period;
}

template<int N>
inline void Saver<N>::setEnsemble(Simulation<N>* e)
{
	my = e;
}

//**********************************************************************************************************************
// Simulation implementation
//**********************************************************************************************************************

template<int N>
Simulation<N>::Simulation(vector<int> num, real temp, vector<real> rCut, real deltat, real len)
{
    // Various initializations
	dt		  = deltat;
	L		  = len;
	
	//savers.clear();
	step = 0;
	
	x0   = -L/2;
	y0   = -L/2;
	z0   = -L/2;
	xmax = L/2;
	ymax = L/2;
	zmax = L/2;
	
    for (int type=0; type<N; type++)
    {
        part[type]	  = new Particles;
        part[type]->n = num[type];
        _allocate(part[type]);
    }

    // Initial conditions for positions and velocities
	mt19937 gen;
    uniform_real_distribution<real> u1(0, 0.25*L);
    uniform_real_distribution<real> u0(0.4*L, 0.5*L);
    uniform_real_distribution<real> uphi(0, 2*M_PI);
    uniform_real_distribution<real> utheta(0, M_PI);

    
    if (N>1) setLattice(part[1]->x, part[1]->y, part[1]->z, 7, 3.5, 7, part[1]->n);

    
//    for (int i=0; i<num[0]; i++)
//    {
//        real r = u0(gen);
//        real phi = uphi(gen);
//        real theta = utheta(gen);
//        
//        part[0]->x[i] = r * sin(theta) * cos(phi);
//        part[0]->y[i] = r * sin(theta) * sin(phi);
//        part[0]->z[i] = r * cos(theta);
//    }

    setLattice(part[0]->x, part[0]->y, part[0]->z, L, L, L, part[0]->n);

    // Initialize cell list if we need it
#ifdef MD_USE_CELLLIST
    real lower[3]  = {x0,   y0,   z0};
    real higher[3] = {xmax, ymax, zmax};
    
    for (int type=0; type<N; type++)
        cells[type] = new Cells<Particles>(part[type], part[type]->n, rCut[type], lower, higher);
#endif
    
    for (int type=0; type<N; type++)
    {
        _fill(part[type]->vx, part[type]->n, 0.0);
        _fill(part[type]->vy, part[type]->n, 0.0);
        _fill(part[type]->vz, part[type]->n, 0.0);
        
        _fill(part[type]->m, part[type]->n, 1.0);
        _fill(part[type]->label, part[type]->n, 0);
    }
    
    //_fill(part[0]->vz, part[0]->n, 0.5);
    
    Forces::Arguments::part = part;
    Forces::Arguments::L = L;
#ifdef MD_USE_CELLLIST
    Forces::Arguments::cells = cells;
#endif
}

template<int N>
void Simulation<N>::setLattice(real *x, real *y, real *z, real lx, real ly, real lz, int n)
{
	real h = pow(lx*ly*lz/n, 1.0/3);
    int nx = ceil(lx/h);
    int ny = ceil(ly/h);
    int nz = ceil((real)n/(nx*ny));
	
	int i=1;
	int j=1;
	int k=1;
	
	for (int tot=0; tot<n; tot++)
	{
		x[tot] = i*h - lx/2;
		y[tot] = j*h - ly/2;
		z[tot] = k*h - lz/2;
		
		i++;
		if (i > nx)
		{
			j++;
			i = 1;
		}
		
		if (j > ny)
		{
			k++;
			j = 1;
		}
	}
}

template<int N>
real Simulation<N>::Ktot(int type)
{
    real res = 0;
    
    if (type < 0)
    {
        for (int t=0; t<N; t++)
            res += _kineticNrg(part[t]);
    }
    else
    {
        res = _kineticNrg(part[type]);
    }
	return res;
}

template<int N>
void Simulation<N>::linMomentum(real& px, real& py, real& pz, int type)
{
    real _px = 0, _py = 0, _pz = 0;
    px = py = pz = 0;
    
	if (type < 0)
    {
        for (int t=0; t<N; t++)
        {
            _linMomentum(part[t], _px, _py, _pz);
            px += _px;
            py += _py;
            pz += _pz;
        }
    }
    else
    {
        _linMomentum(part[type], px, py, pz);
    }
}

template<int N>
void Simulation<N>::angMomentum(real& Lx, real& Ly, real& Lz, int type)
{
    real _Lx = 0, _Ly = 0, _Lz = 0;
    Lx = Ly = Lz = 0;
    
	if (type < 0)
    {
        for (int t=0; t<N; t++)
        {
            _angMomentum(part[t], _Lx, _Ly, _Lz);
            Lx += _Lx;
            Ly += _Ly;
            Lz += _Lz;
        }
    }
    else
    {
        _angMomentum(part[type], Lx, Ly, Lz);
    }
}

template<int N>
void Simulation<N>::centerOfMass(real& mx, real& my, real& mz, int type)
{
    real _mx = 0, _my = 0, _mz = 0;
    mx = my = mz = 0;
    
	if (type < 0)
    {
        for (int t=0; t<N; t++)
        {
            _centerOfMass(part[t], _mx, _my, _mz);
            mx += _mx;
            my += _my;
            mz += _mz;
        }
    }
    else
    {
        _centerOfMass(part[type], mx, my, mz);
    }
}


inline void lowbound(real* x, real* v, int n, real dt, real lim)
{
    for (int i=0; i<n; i++)
        if (x[i] < lim)
        {
            x[i] = lim + lim - x[i];
            v[i] = -v[i];
        }
}

inline void highbound(real* x, real* v, int n, real dt, real lim)
{
    for (int i=0; i<n; i++)
        if (x[i] > lim)
        {
            x[i] = lim - (x[i] - lim);
            v[i] = -v[i];
        }
}

inline void addForce(real* x, real* ax, int* labels, int n, real F, int label)
{
    real totF = 0;
    int num = 0;
    
    for (int i=0; i<n; i++)
        if (labels[i] == label)
        {
            num++;
            totF += ax[i];
        }
    
    real appliedF = totF / num + F;
    
    for (int i=0; i<n; i++)
        if (labels[i] == label)
            ax[i] = appliedF;
}

inline void fix(real* a, int* labels, int n, real dt, int label)
{
    for (int i=0; i<n; i++)
        if (labels[i] == label)
            a[i] = 0;
}

template<int N>
void Simulation<N>::velocityVerlet()
{
    static bool fixed = false;
    
    const real begin = 200;
    
    auto prep = [&](int type)
    {
        _axpy(part[type]->vx, part[type]->ax,part[type]->n, dt*0.5);
        _axpy(part[type]->vy, part[type]->ay,part[type]->n, dt*0.5);
        _axpy(part[type]->vz, part[type]->az,part[type]->n, dt*0.5);
        
        _axpy(part[type]->x, part[type]->vx,part[type]->n, dt);
        _axpy(part[type]->y, part[type]->vy,part[type]->n, dt);
        _axpy(part[type]->z, part[type]->vz,part[type]->n, dt);
        
        _normalize(part[type]->x,part[type]->n, x0, xmax);
        _normalize(part[type]->y,part[type]->n, y0, ymax);
        _normalize(part[type]->z,part[type]->n, z0, zmax);
        
        _fill(part[type]->ax,part[type]->n, 0.0);
        _fill(part[type]->ay,part[type]->n, 0.0);
        _fill(part[type]->az,part[type]->n, 0.0);
    };
    
    profiler.start("K1");
    Unroller<0, N>::step(prep);
    profiler.stop();
    
    if (dt*step < begin)
    {
        lowbound(part[1]->y, part[1]->vy, part[1]->n, dt,  -1.2);
        highbound(part[1]->y, part[1]->vy, part[1]->n, dt, 2.2);
    }
    else
    {
        if (!fixed)
        {
            for (int i=0; i<part[1]->n; i++)
            {
                if (part[1]->y[i] > 1.9) part[1]->label[i] = 2;
                if (part[1]->y[i] < -0.9) part[1]->label[i] = 1;
            }
            
            fix(part[1]->vx, part[1]->label, part[1]->n, dt,  1);
            fix(part[1]->vy, part[1]->label, part[1]->n, dt,  1);
            fix(part[1]->vz, part[1]->label, part[1]->n, dt,  1);
            
            fix(part[1]->vx, part[1]->label, part[1]->n, dt,  2);
            fix(part[1]->vy, part[1]->label, part[1]->n, dt,  2);
            fix(part[1]->vz, part[1]->label, part[1]->n, dt,  2);
            
            fixed = true;
        }
    }


#ifdef MD_USE_CELLLIST
    auto docells = [&](int type)
    {
        cells[type]->migrate();
    };
    
    profiler.start("CellList");
    Unroller<0, N>::step(docells);
    profiler.stop();
#endif
    
    profiler.start("K2");
    Unroller2<0, N, 0, N>::step();
    profiler.stop();
    
    if (dt*step >= begin)
    {
        fix(part[1]->ax, part[1]->label, part[1]->n, dt,  2);
        //fix(part[1]->ay, part[1]->label, part[1]->n, dt,  2);
        addForce(part[1]->y, part[1]->ay, part[1]->label, part[1]->n, 10, 2);
        fix(part[1]->az, part[1]->label, part[1]->n, dt,  2);
        
        fix(part[1]->ax, part[1]->label, part[1]->n, dt,  1);
        fix(part[1]->ay, part[1]->label, part[1]->n, dt,  1);
        fix(part[1]->az, part[1]->label, part[1]->n, dt,  1);
    }
    
//    for (int i=0; i<part[0]->n; i++)
//        part[0]->az[i] += (part[0]->x[i] > 0) ? 0.02 : -0.02;

    auto fin = [&](int type)
    {
        _nuscal(part[type]->ax, part[type]->m, part[type]->n);
        _nuscal(part[type]->ay, part[type]->m, part[type]->n);
        _nuscal(part[type]->az, part[type]->m, part[type]->n);
        
        _axpy(part[type]->vx, part[type]->ax, part[type]->n, dt*0.5);
        _axpy(part[type]->vy, part[type]->ay, part[type]->n, dt*0.5);
        _axpy(part[type]->vz, part[type]->az, part[type]->n, dt*0.5);
        
    };
    
    Unroller<0, N>::step(fin);
}

template<int N>
void Simulation<N>::runOneStep()
{
    profiler.start("Other");
	for (typename list<Saver<N>*>::iterator it = savers.begin(), end = savers.end(); it != end; ++it)
	{
		if ((step % (*it)->getPeriod()) == 0) (*it)->exec();
	}
    profiler.stop();
	
	if (step == 0)
        Unroller2<0, N, 0, N>::step();

	step++;
	velocityVerlet();
}

template<int N>
void Simulation<N>::registerSaver(Saver<N>* saver, int period)
{
    if (period > 0)
    {
        saver->setEnsemble(this);
        saver->setPeriod(period);
        savers.push_back(saver);
    }
}







