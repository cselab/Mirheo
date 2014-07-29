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
//#include "Simulation.h"
#include "Misc.h"
#include "Particles.h"

using namespace std;

class Simulation;

class Saver
{
public:
    static string folder;
    
protected:
	int period;
	ofstream* file;
	Simulation* my;
    
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
	
	inline  void setEnsemble(Simulation* e) { my = e; }
	inline  void setPeriod(int p)           { period = p; }
	inline  int  getPeriod()                { return period; }
	
	inline static bool makedir(string name)
	{
		folder = name;
		if (mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 && errno != EEXIST) return false;
		return true;
	}
	
	virtual void exec() = 0;
};


class SaveEnergy: public Saver
{
public:
	SaveEnergy(ostream* o)		  : Saver(o) {};
	SaveEnergy(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		//real V = my->Vtot();
        real V = 0;
		real K = this->my->Ktot();
		(*this->file) << V << " " << K << " " << V+K << endl;
        this->file->flush();
	}
};

class SaveTemperature: public Saver
{
public:
	SaveTemperature(ostream* o)		  : Saver(o) {};
	SaveTemperature(string fname)	  : Saver(fname) {};
	
	void exec()
	{
        int tot = 0;
        int N = this->my->getNTypes();

        for (int type = 0; type < N; type++)
            tot += this->my->getParticles()[type]->n;
        
		real K = this->my->Ktot();
		(*this->file) << K * 0.6666666666666666 / tot  << endl;
        this->file->flush();
	}
};

class SaveLinMom: public Saver
{
public:
	SaveLinMom(ostream* o)		  : Saver(o) {};
	SaveLinMom(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		real px, py, pz;
		this->my->linMomentum(px, py, pz);
		(*this->file) << px << " " << py << " " << pz << endl;
        this->file->flush();
	}
};

class SaveAngMom: public Saver
{
public:
	SaveAngMom(ostream* o)		  : Saver(o) {};
	SaveAngMom(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		real Lx, Ly, Lz;
		this->my->angMomentum(Lx, Ly, Lz);
		(*this->file) << Lx << " " << Ly << " " << Lz << endl;
	}	
};

class SaveCenterOfMass: public Saver
{
public:
	SaveCenterOfMass(ostream* o)	  : Saver(o) {};
	SaveCenterOfMass(string fname)	  : Saver(fname) {};
	
	void exec()
	{
		real Mx, My, Mz;
		this->my->centerOfMass(Mx, My, Mz);
		(*this->file) << Mx << " " << My << " " << Mz << endl;
	}	
};

class SavePos: public Saver
{
public:
	SavePos(ostream* o)		   : Saver(o) {};
	SavePos(string fname)	   : Saver(fname) {};
	
	void exec()
	{
        static const char* atoms = "HONCBFPSKYI";
		Particles** p = this->my->getParticles();
		
        int tot = 0;
        int N = this->my->getNTypes();

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

class SaveStrain: public Saver
{
public:
	SaveStrain(ostream* o)		   : Saver(o) {};
	SaveStrain(string fname)	   : Saver(fname) {};
	
	void exec()
	{
        real diff = fabs(this->my->walls[0]->z - this->my->walls[1]->z);
        
		*this->file << this->my->getIter() << " " << this->my->walls[0]->z << " " << this->my->walls[1]->z << " " << diff << endl;
        this->file->flush();
	}
};


class SaveRestart: public Saver
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
        
        int N = this->my->getNTypes();
        out.write((char*)&N, sizeof(int));
        
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

class SaveTiming: public Saver
{
private:
	Timer t;
	
public:
	SaveTiming(ostream* o)		  : Saver(o) {};
	SaveTiming(string fname)	  : Saver(fname) {};
	
	
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


class SaveRdf: public Saver
{
private:
	Timer t;
    real* bins;
    real* totBins;
    int nBins;
    real h;
    int total;
    int prnPeriod;
	
public:
	SaveRdf(ostream* o, int prnPeriod) : Saver(o), prnPeriod(prnPeriod)
    {
        h = 0.005;
        nBins = 2.0 / h;
        bins = new real [nBins];
        totBins = new real [nBins];
        for (int i=0; i<nBins; i++)
            totBins[i] = 0;
        total = 0;
    };
    
	SaveRdf(string fname, int prnPeriod) : Saver(fname), prnPeriod(prnPeriod)
    {
        h = 0.005;
        nBins = 2.0 / h;
        bins = new real [nBins];
        totBins = new real [nBins];
        for (int i=0; i<nBins; i++)
            totBins[i] = 0;
        total = 0;
    };
    
	void exec()
	{
        this->my->rdf(1, bins, h, nBins);
        total++;
        
        
        for (int i=0; i<nBins; i++)
            totBins[i] += bins[i];
        
        if ( (this->my->getIter() % prnPeriod) == 0 )
        {
            *this->file << "Iteration  " << this->my->getIter() << endl;
            for (int i=0; i<nBins; i++)
                *this->file << i*h << "  " << totBins[i]/total << endl;
            
            *this->file << endl;
            this->file->flush();
        }
	}
};








