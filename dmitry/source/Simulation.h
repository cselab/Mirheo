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

#include "Profiler.h"
#include "Potential.h"
#include "timer.h"
#include "Misc.h"
#ifdef MD_USE_CELLLIST
#include "CellList.h"
#endif

using namespace std;

class Saver;

//**********************************************************************************************************************
// Particles
//
// Structure of arrays is used for efficient the computations
//**********************************************************************************************************************
struct Particles
{
	double *x,  *y,  *z;
	double *vx, *vy, *vz;
	double *ax, *ay, *az;
	double *m;
	double *tmp;
    double *type;
	int n;
};

class Simulation
{
private:
	int    n;				// just for convenience
	
	int    step;
	double x0, y0, z0, xmax, ymax, zmax;
	double dt;
	double L;
	
	Particles*		part;
	DPD*	potential;
	list<Saver*>	savers;
	
#ifdef MD_USE_CELLLIST
	Cells<Particles>* cells;
#endif
	
	Randomer rng;
	
	void velocityVerlet();
	
public:
	Profiler profiler;

	Simulation (int, double, double, double, double, double, double, double);
	void setLattice(double *x, double *y, double *z);
	
	double Ktot();
	double Vtot();
	double Etot();
	void linMomentum (double& px, double& py, double& pz);
	void angMomentum (double& Lx, double& Ly, double& Lz);
	void centerOfMass(double& mx, double& my, double& mz);
	
	inline int getIter() { return step; }
	inline Particles* getParticles() { return part; }
	
	void getPositions(int&, double*, double*);
	void registerSaver(Saver*, int);
	void runOneStep();
};



class Saver
{
protected:
	int period;
	ofstream* file;
	static string folder;
	Simulation* my;
    
    bool opened;
	
public:
	Saver (ostream* cOut) : file((ofstream*)cOut) { opened = false; }
	Saver (string fname)
	{
		file = new ofstream((this->folder+fname).c_str());
        opened = true;
	}
	Saver (){};
    ~Saver () { if (!opened) file->close(); }
	
	inline  void setEnsemble(Simulation* e);
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

inline void Saver::setPeriod(int p)
{
	period = p;
}

inline int Saver::getPeriod()
{
	return period;
}

inline void Saver::setEnsemble(Simulation* e)
{
	my = e;
}






