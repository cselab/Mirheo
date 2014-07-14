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

#ifdef __CUDACC__
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
double Forces::Arguments::L;
#ifdef MD_USE_CELLLIST
Cells<Particles>** Forces::Arguments::cells;
#endif

template<int N>
class Simulation
{
private:
	int    step;
	double x0, y0, z0, xmax, ymax, zmax;
	double dt;
	double L;
	
	Particles* part[N];
	list<Saver<N>*>	savers;
	
#ifdef MD_USE_CELLLIST
	Cells<Particles>* cells[N];
#endif
		
	void velocityVerlet();
	
public:
	Profiler profiler;

	Simulation (vector<int>, double, double, double, double);
	void setLattice(double *x, double *y, double *z, double l, int n);
	
	double Ktot(int = -1);
	void linMomentum (double& px, double& py, double& pz, int = -1);
	void angMomentum (double& px, double& py, double& pz, int = -1);
	void centerOfMass(double& mx, double& my, double& mz, int = -1);
	
	inline int getIter() { return step; }
	inline Particles** getParticles() { return part; }
	
	void getPositions(int&, double*, double*);
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
Simulation<N>::Simulation(vector<int> num, double temp, double rCut, double deltat, double len)
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
    uniform_real_distribution<double> u1(0, 0.25*L);
    uniform_real_distribution<double> u0(0.4*L, 0.5*L);
    uniform_real_distribution<double> uphi(0, 2*M_PI);
    uniform_real_distribution<double> utheta(0, M_PI);

    
    setLattice(part[1]->x, part[1]->y, part[1]->z, 3.04, part[1]->n);
    
    for (int i=0; i<num[0]; i++)
    {
        double r = u0(gen);
        double phi = uphi(gen);
        double theta = utheta(gen);
        
        part[0]->x[i] = r * sin(theta) * cos(phi);
        part[0]->y[i] = r * sin(theta) * sin(phi);
        part[0]->z[i] = r * cos(theta);
    }

    
    if (N > 10)
        for (int i=0; i<num[1]; i++)
        {
            double r = u1(gen);
            double phi = uphi(gen);
            double theta = utheta(gen);
            
            part[1]->x[i] = r * sin(theta) * cos(phi);
            part[1]->y[i] = r * sin(theta) * sin(phi);
            part[1]->z[i] = r * cos(theta);
        }


    // Initialize cell list if we need it
#ifdef MD_USE_CELLLIST
    double lower[3]  = {x0,   y0,   z0};
    double higher[3] = {xmax, ymax, zmax};
    
    for (int type=0; type<N; type++)
        cells[type] = new Cells<Particles>(part[type], part[type]->n, rCut, lower, higher);
#endif
    
    for (int type=0; type<N; type++)
    {
        _fill(part[type]->vx, part[type]->n, 0);
        _fill(part[type]->vy, part[type]->n, 0);
        _fill(part[type]->vz, part[type]->n, 0);
        
        _fill(part[type]->m, part[type]->n, 1);
    }
    
    //_fill(part[0]->vz, part[0]->n, 0.5);
    
    Forces::Arguments::part = part;
    Forces::Arguments::L = L;
#ifdef MD_USE_CELLLIST
    Forces::Arguments::cells = cells;
#endif
}

template<int N>
void Simulation<N>::setLattice(double *x, double *y, double *z, double l, int n)
{
	int nAlongDim = ceil(pow(n, 1.0/3.0));
	double h = l / (nAlongDim + 1);
	
	int i=1;
	int j=1;
	int k=1;
	
	for (int tot=0; tot<n; tot++)
	{
		x[tot] = i*h - l/2;
		y[tot] = j*h - l/2;
		z[tot] = k*h - l/2;
		
		i++;
		if (i > nAlongDim)
		{
			j++;
			i = 1;
		}
		
		if (j > nAlongDim)
		{
			k++;
			j = 1;
		}
	}
}

template<int N>
double Simulation<N>::Ktot(int type)
{
    double res = 0;
    
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
void Simulation<N>::linMomentum(double& px, double& py, double& pz, int type)
{
    double _px = 0, _py = 0, _pz = 0;
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
void Simulation<N>::angMomentum(double& Lx, double& Ly, double& Lz, int type)
{
    double _Lx = 0, _Ly = 0, _Lz = 0;
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
void Simulation<N>::centerOfMass(double& mx, double& my, double& mz, int type)
{
    double _mx = 0, _my = 0, _mz = 0;
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

template<int N>
void Simulation<N>::velocityVerlet()
{
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

        _fill(part[type]->ax,part[type]->n, 0);
        _fill(part[type]->ay,part[type]->n, 0);
        _fill(part[type]->az,part[type]->n, 0);
    };
    
    profiler.start("K1");
    Unroller<0, N>::step(prep);
    profiler.stop("K1");

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
    
    for (int i=0; i<part[0]->n; i++)
        part[0]->az[i] += (part[0]->x[i] > 0) ? 0.02 : -0.02;

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
	for (typename list<Saver<N>*>::iterator it = savers.begin(), end = savers.end(); it != end; ++it)
	{
		if ((step % (*it)->getPeriod()) == 0) (*it)->exec();
	}
	
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







