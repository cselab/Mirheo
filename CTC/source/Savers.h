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
#include "Misc.h"
#include "Particles.h"

using namespace std;

template<int N>
class SaveEnergy: public Saver<N>
{
public:
	SaveEnergy(ostream* o)		  : Saver<N>(o) {};
	SaveEnergy(string fname)	  : Saver<N>(fname) {};
	
	void exec()
	{
		//real V = my->Vtot();
        real V = 0;
		real K = this->my->Ktot();
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
        
		real K = this->my->Ktot();
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
		real px, py, pz;
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
		real Lx, Ly, Lz;
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
		real Mx, My, Mz;
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
		Particles** p = this->my->getParticles();
		
        int tot = 0;
        for (int type = 0; type < N; type++)
            tot += p[type]->n;
        
		*this->file << tot << endl;
		*this->file << "MD simulation" << endl;
		
        for (int type = 0; type < N; type++)
            for (int i = 0; i < p[type]->n; i++)
                *this->file << atoms[type] << " " << p[type]->x(i) << " " << p[type]->y(i) << " " << p[type]->z(i) << endl;
		this->file->flush();
	}
};

template<int N>
class SaveRestart: public Saver<N>
{
    string fname;
    int spaceOddity;
    bool continuous;
    
public:
	SaveRestart (string fname, bool continuous = false) :
        fname(fname), spaceOddity(0), continuous(continuous) {};
    
	void exec()
	{
        Particles** p = this->my->getParticles();
        ofstream out((this->folder + fname + to_string(spaceOddity)).c_str(), ios::out | ios::binary);
		
        if (continuous)
            spaceOddity++;
        else
            spaceOddity = spaceOddity ? 0 : 1;
        
        int n = N;
        out.write((char*)&n, sizeof(int));
        
        for (int i = 0; i<N; i++)
        {
            out.write((char*)&p[i]->n, sizeof(int));
            
            out.write((char*)p[i]->xdata,  3*p[i]->n*sizeof(real));
            out.write((char*)p[i]->vdata,  3*p[i]->n*sizeof(real));
            
            out.write((char*)p[i]->m,  p[i]->n*sizeof(real));
            
            out.write((char*)p[i]->label, p[i]->n*sizeof(int));
        }
        out.close();
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









