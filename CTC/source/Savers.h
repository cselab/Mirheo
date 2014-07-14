/*
 *  Saver.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "timer.h"
#include "Simulation.h"

using namespace std;

template<int N>
class SaveEnergy: public Saver<N>
{
public:
	SaveEnergy(ostream* o)		  : Saver<N>(o) {};
	SaveEnergy(string fname)	  : Saver<N>(fname) {};
	
	void exec()
	{
		//double V = my->Vtot();
        double V = 0;
		double K = this->my->Ktot();
		(*this->file) << V << " " << K << " " << V+K << endl;
        this->file->flush();
	}
};

template<int N>
class SaveTemperature: public Saver<N>
{
public:
	SaveTemperature(ostream* o)		  : Saver<N>(o) {};
	SaveTemperature(string fname)	  : Saver<N>(fname) {};
	
	void exec()
	{
        int tot = 0;
        for (int type = 0; type < N; type++)
            tot += this->my->getParticles()[type]->n;
        
		double K = this->my->Ktot();
		(*this->file) << K * 0.6666666666666666 / tot  << endl;
        this->file->flush();
	}
};

template<int N>
class SaveLinMom: public Saver<N>
{
public:
	SaveLinMom(ostream* o)		  : Saver<N>(o) {};
	SaveLinMom(string fname)	  : Saver<N>(fname) {};
	
	void exec()
	{
		double px, py, pz;
		this->my->linMomentum(px, py, pz);
		(*this->file) << px << " " << py << " " << pz << endl;
        this->file->flush();
	}
};

template<int N>
class SaveAngMom: public Saver<N>
{
public:
	SaveAngMom(ostream* o)		  : Saver<N>(o) {};
	SaveAngMom(string fname)	  : Saver<N>(fname) {};
	
	void exec()
	{
		double Lx, Ly, Lz;
		this->my->angMomentum(Lx, Ly, Lz);
		(*this->file) << Lx << " " << Ly << " " << Lz << endl;
	}	
};

template<int N>
class SaveCenterOfMass: public Saver<N>
{
public:
	SaveCenterOfMass(ostream* o)	  : Saver<N>(o) {};
	SaveCenterOfMass(string fname)	  : Saver<N>(fname) {};
	
	void exec()
	{
		double Mx, My, Mz;
		this->my->centerOfMass(Mx, My, Mz);
		(*this->file) << Mx << " " << My << " " << Mz << endl;
	}	
};

template<int N>
class SavePos: public Saver<N>
{
public:
	SavePos(ostream* o)		   : Saver<N>(o) {};
	SavePos(string fname)	   : Saver<N>(fname) {};
	
	void exec()
	{
        static const char* atoms = "HONCBFPSKYI";
#ifndef __CUDACC__
		Particles** p = this->my->getParticles();
		
        int tot = 0;
        for (int type = 0; type < N; type++)
            tot += p[type]->n;
        
		*this->file << tot << endl;
		*this->file << "MD simulation" << endl;
		
        for (int type = 0; type < N; type++)
            for (int i = 0; i < p[type]->n; i++)
                *this->file << atoms[type] << " " << p[type]->x[i] << " " << p[type]->y[i] << " " << p[type]->z[i] << endl;
		this->file->flush();
#endif
	}
};

template<int N>
class SaveTiming: public Saver<N>
{
private:
	Timer t;
	
public:
	SaveTiming(ostream* o)		  : Saver<N>(o) {};
	SaveTiming(string fname)	  : Saver<N>(fname) {};
	
	
	void exec()
	{
		if (this->my->getIter() < this->period) t.start();
		else
		{
			*this->file << this->my->profiler.printStat() << endl;
            this->file->flush();
		}
	}
};









