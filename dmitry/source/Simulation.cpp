/*
 *  Simulation.cpp
 *  hpchw
 *
 *  Created by Dmitry Alexeev on 17.10.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Simulation.h"

#ifdef __CUDACC__
#include "Gpukernels.h"
#else
#include "Cpukernels.h"
#endif

Simulation::Simulation(int num, double temp, double mass, double alpha, double gamma, double rCut, double deltat, double len)
{
// Various initializations
	
	potential = new DPD(alpha, gamma, temp, rCut, deltat);
	part	  = new Particles;
	dt		  = deltat;
	L		  = len;
	
	savers.clear();
	step = 0;
	
	x0   = -L/2;
	y0   = -L/2;
	z0   = -L/2;
	xmax = L/2;
	ymax = L/2;
	zmax = L/2;
	
	part->n = n = num;
	_allocate(part);
	
// Initial conditions for positions and velocities
	double V, K;
	double *vx, *vy, *vz, *x, *y, *z;

#ifdef __CUDACC__
	x  = new double[n];
	y  = new double[n];
	z  = new double[n];
	vx = new double[n];
	vy = new double[n];
	vz = new double[n];
#else
	x  = part->x;
	y  = part->y;
	z  = part->z;
	vx = part->vx;
	vy = part->vy;
	vz = part->vz;
#endif
	
	setLattice(x, y, z);
	
#ifdef __CUDACC__
	_copy(x, part->x, n);
	_copy(y, part->y, n);
	_copy(z, part->z, n);
#endif


// Initialize cell list if we need it
#ifdef MD_USE_CELLLIST
    double lower[3]  = {x0,   y0,   z0};
    double higher[3] = {xmax, ymax, zmax};
	cells = new Cells<Particles>(part, n, rCut, lower, higher);
#endif
		
	V = Vtot();
	for (int i = 0; i<n; i++)
	{
		vx[i] = rng.getNormalRand();
		vy[i] = rng.getNormalRand();
		vz[i] = rng.getNormalRand();
	}
	_fill(part->m, n, mass);

#ifdef __CUDACC__
	_copy(vx, part->vx, n);
	_copy(vy, part->vy, n);
	_copy(vz, part->vz, n);
#endif

//	K = energy - V;
//	double alpha = K / Ktot();
//	
//	if (alpha < 0) alpha = 0;
//	else		   alpha = sqrt(alpha);
//	
//	_scal(part->vx, n, alpha);
//	_scal(part->vy, n, alpha);
//	_scal(part->vz, n, alpha);
}

void Simulation::setLattice(double *x, double *y, double *z)
{
	int nAlongDim = ceil(pow(n, 1.0/3.0));
	double h = L / (nAlongDim + 1);
	
	int i=1;
	int j=1;
	int k=1;
	
	for (int tot=0; tot<n; tot++)
	{
		x[tot] = i*h + x0;
		y[tot] = j*h + y0;
		z[tot] = k*h + z0;
		
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


double Simulation::Vtot()
{
#ifdef MD_USE_CELLLIST
	return _potentialNrg(part, potential, cells, L);
#else
	return _potentialNrg(part, potential, L);
#endif
}

double Simulation::Ktot()
{
	return _kineticNrg(part);
}

double Simulation::Etot()
{
	return Vtot() + Ktot();
}

void Simulation::linMomentum(double& px, double& py, double& pz)
{
	_linMomentum(part, px, py, pz);
}

void Simulation::angMomentum(double& Lx, double& Ly, double& Lz)
{
	_angMomentum(part, Lx, Ly, Lz);
}

void Simulation::centerOfMass(double& mx, double& my, double& mz)
{
	_centerOfMass(part, mx, my, mz);
}


void Simulation::velocityVerlet()
{
	profiler.start("K1");
	{
		_K1(part->vx, part->ax, n, dt*0.5);
		_K1(part->vy, part->ay, n, dt*0.5);
		_K1(part->vz, part->az, n, dt*0.5);
		
		_K1(part->x, part->vx, n, dt);
		_K1(part->y, part->vy, n, dt);
		_K1(part->z, part->vz, n, dt);
	}
	profiler.stop();

	profiler.start("Normalize");
	{
		_normalize(part->x, n, x0, xmax);
		_normalize(part->y, n, y0, ymax);
		_normalize(part->z, n, z0, zmax);
	}
	profiler.stop();


#ifdef MD_USE_CELLLIST
	profiler.start("CellList");
		cells->migrate();
	profiler.stop();
	
	profiler.start("K2");
		_K2(part, potential, cells, L);
	profiler.stop();
#else
	profiler.start("K2");
	_K2(part, potential, L);
	profiler.stop();
#endif
	
	//profiler.start("K1");
	{
		_K1(part->vx, part->ax, n, dt*0.5);
		_K1(part->vy, part->ay, n, dt*0.5);
		_K1(part->vz, part->az, n, dt*0.5);
	}
	//profiler.stop();
}

void Simulation::runOneStep()
{
	for (list<Saver*>::iterator it = savers.begin(), end = savers.end(); it != end; ++it)
	{
		if ((step % (*it)->getPeriod()) == 0) (*it)->exec();
	}
	
	if (step == 0)
	{
#ifdef MD_USE_CELLLIST
		_K2(part, potential, cells, L);
#else
		_K2(part, potential, L);
#endif
	}
	step++;
	
	velocityVerlet();
}

void Simulation::registerSaver(Saver* saver, int period)
{
	saver->setEnsemble(this);
	saver->setPeriod(period);
	savers.push_back(saver);
}
