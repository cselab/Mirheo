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
#include <assert.h>

#include "Particles.h"
#include "Profiler.h"
#include "Interaction.h"
#include "timer.h"
#include "Misc.h"

using namespace std;

class Saver;

//**********************************************************************************************************************
// Adhesive walls
//**********************************************************************************************************************

struct LJwallz
{
    real z;
    real v;
    real a;
    const real mass = 1;
    
    LJwallz(real z, real v):z(z), v(v), a(0) {};
    
    void VVpre(real dt)
    {
        v += a * dt*0.5;
        z += v * dt;
    }
    
    void VVpost(real dt)
    {
        v += a * dt*0.5;
    }
    
    void force(Particles* part)
    {
        static const real rCut  = 2;
        static const real sigma = 0.15;
        static const real eps   = 0.01;
        
        a = 0;
        for (int i = 0; i<part->n; i++)
        {
            if (fabs(part->z(i) - z) < rCut)
            {
                real r = fabs(part->z(i) - z);
                real s_r = sigma / r;
                real s_r2 = s_r*s_r;
                real f5 = s_r*s_r2*s_r2; // 5
                
                real f11 = s_r*f5*f5;    // 11
                
                real f = eps * 1/(sigma*sigma) * (3*f5 - f11) * (part->z(i) - z);
                part->az(i) -= f / part->m[i];
                a += f / mass;
                a -= 0.01*v;
                
                if (f > 100) info("wall: %.5f,   part %.5f,   r: %.4f   f5: %.4f  f11: %.4f\n", z, part->z(i), r, 3*f5, f11);
            }
        }
        
        //info("%.5f   %.5f   %.5f\n", z, v, a);
    }
    
    void fix()
    {
        a = 0;
        v = 0;
    }
    
    void addF(real f)
    {
        a += f/mass;
    }
    
};


//**********************************************************************************************************************
// Simulation
//**********************************************************************************************************************
class Simulation
{
private:
    
    int nTypes;
    
	int  step;
	real xlo, ylo, zlo, xhi, yhi, zhi;
	real dt;
	
	Particles** part;
	list<Saver*>	savers;
    
    InteractionTable* evaluator;
		
	void velocityVerlet();
    
public:
	Profiler profiler;
    
    vector<LJwallz*> walls;

	Simulation (int, real);
	void setLattice(Particles* p, real lx, real ly, real lz, int n);
    void setIC();
    void loadRestart(string fname);
	
	real Ktot(int = -1);
	void linMomentum (real& px, real& py, real& pz, int = -1);
	void angMomentum (real& px, real& py, real& pz, int = -1);
	void centerOfMass(real& mx, real& my, real& mz, int = -1);
	
	inline int getIter()   { return step; }
    inline int getNTypes() { return nTypes; }
	inline Particles** getParticles() { return part; }
	
	void getPositions(int&, real*, real*);
	void registerSaver(Saver*, int);
	void runOneStep();
};






