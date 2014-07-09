/*
 *  Saver.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Simulation.h"
#include "timer.h"

using namespace std;

class SaveEnergy: public Saver
{
public:
	SaveEnergy(ostream* o)		  : Saver(o) {};
	SaveEnergy(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		double V = my->Vtot();
		double K = my->Ktot();
		(*file) << V << " " << K << " " << V+K << endl;
        file->flush();
	}
};

class SaveTemperature: public Saver
{
public:
	SaveTemperature(ostream* o)		  : Saver(o) {};
	SaveTemperature(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		double K = my->Ktot();
		(*file) << K * 0.6666666666666666 / my->getParticles()->n << endl;
	}
};

class SaveLinMom: public Saver
{
public:
	SaveLinMom(ostream* o)		  : Saver(o) {};
	SaveLinMom(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		double px, py, pz;
		my->linMomentum(px, py, pz);
		(*file) << px << " " << py << " " << pz << endl;
	}	
};

class SaveAngMom: public Saver
{
public:
	SaveAngMom(ostream* o)		  : Saver(o) {};
	SaveAngMom(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		double Lx, Ly, Lz;
		my->angMomentum(Lx, Ly, Lz);
		(*file) << Lx << " " << Ly << " " << Lz << endl;
	}	
};

class SaveCenterOfMass: public Saver
{
public:
	SaveCenterOfMass(ostream* o)	  : Saver(o) {};
	SaveCenterOfMass(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		double Mx, My, Mz;
		my->centerOfMass(Mx, My, Mz);
		(*file) << Mx << " " << My << " " << Mz << endl;
	}	
};

class SavePos: public Saver
{
public:
	SavePos(ostream* o)		   : Saver(o) {};
	SavePos(string fname)	   : Saver(fname) {};
	
	void exec()
	{
#ifndef __CUDACC__
		Particles* p = my->getParticles();
		
		*file << p->n << endl;
		*file << "MD simulation" << endl;
		
		for (int i = 0; i < p->n; i++)
			*file << "H " << p->x[i] << " " << p->y[i] << " " << p->z[i] << endl;
		file->flush();
#endif
	}
};

class SaveTiming: public Saver
{
private:
	Timer t;
	
public:
	SaveTiming(ostream* o)		  : Saver(o) {};
	SaveTiming(string fname)	  : Saver(fname) {};
	
	
	void exec()
	{
		if (my->getIter() < period) t.start();
		else
		{
			*file << my->profiler.printStat() << endl;
		}
	}
};










