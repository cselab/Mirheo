//
//  Interaction.cpp
//  CTC
//
//  Created by Dmitry Alexeev on 23.07.14.
//  Copyright (c) 2014 Dmitry Alexeev. All rights reserved.
//

#include "Interaction.h"
#include "CellList.h"

#ifdef __GPU__
#include "Gpukernels.h"
#else
#include "Cpukernels.h"
#endif


InteractionTable::InteractionTable(int nTypes, Particles** part, Profiler& prof, real xlo, real xhi,  real ylo, real yhi,  real zlo, real zhi) :
nTypes(nTypes), part(part), prof(prof)
{
    real lower[3]  = {xlo, ylo, zlo};
    real higher[3] = {xhi, yhi, zhi};
    
    rCut  = new real*         [nTypes];
    table = new Interaction** [nTypes];
    types = new InterTypes*   [nTypes];
    for (int i=0; i<nTypes; i++)
    {
        rCut[i]  = new real        [nTypes];
        table[i] = new Interaction*[nTypes];
        types[i] = new InterTypes  [nTypes];
    }
    
    for (int i=0; i<nTypes; i++)
        for (int j=i; j<nTypes; j++)
        {
            string tag = to_string(i+1)+"-"+to_string(j+1);
            string inttag = "Interaction " + tag;
            
            rCut[i][j] = configParser->getf("rCut", tag);
            string type = configParser->gets(inttag, "type");
            
            if (type == "dpd")
            {
                const real aij   = configParser->getf(inttag,  "aij",    2.5);
                const real gamma = configParser->getf(inttag,  "gamma",  45);
                const real temp  = configParser->getf("Basic", "temp",   0.1);
                const real rc    = configParser->getf(inttag,  "rCut",   1);
                const real dt    = configParser->getf("Basic", "dt",     0.001);
                
                table[i][j] = new DPD(rCut[i][j], aij, gamma, temp, rc, dt);
                types[i][j] = INTER_DPD;
            }
            else if (type == "sem")
            {
                const real gamma = configParser->getf(inttag,  "gamma",  100);
                const real temp  = configParser->getf("Basic", "temp",   0.1);
                const real rc    = configParser->getf(inttag,  "rCut",   1);
                const real dt    = configParser->getf("Basic", "dt",     0.001);
                const real u0    = configParser->getf(inttag,  "u0",     0.14);
                const real rho   = configParser->getf(inttag,  "rho",    0.5);
                const real req   = configParser->getf(inttag,  "req",    0);
                const real D     = configParser->getf(inttag,  "D",      1e-5);
                
                table[i][j] = new SEM (rCut[i][j], gamma, temp, rc, dt,  u0, rho, req, D);
                types[i][j] = INTER_SEM;
            }
            else die("Unsoported (yet) interaction type \"%s\"!", type.c_str());
        }
    
    cells = new Cells<Particles>** [nTypes];
    for (int i=0; i<nTypes; i++)
    {
        cells[i] = new Cells<Particles>* [nTypes];
        for (int j=0; j<nTypes; j++)
            cells[i][j] = new Cells<Particles>(part[i], part[i]->n, rCut[min(i, j)][max(i, j)], lower, higher);
    }
}

void InteractionTable::doCells()
{
#ifndef __GPU__
    for (int i=0; i<nTypes; i++)
        for (int j=0; j<nTypes; j++)
            cells[i][j]->migrate();
#endif
}

void InteractionTable::evalForces(int step)
{
    for (int i=0; i<nTypes; i++)
        for (int j=i; j<nTypes; j++)
        {
            // I don't want virtual functions here
            prof.start("Forces " + to_string(i)+" <-> "+to_string(j));
            
            switch (types[i][j])
            {
                case INTER_DPD:
                {
                    DPD* dpd = static_cast<DPD*>(table[i][j]);
                    computeForces(part, cells, i, j, *dpd, step);
                }
                    break;
                    
                case INTER_SEM:
                {
                    SEM* sem = static_cast<SEM*>(table[i][j]);
                    computeForces(part, cells, i, j, *sem, step);
                }
                    break;
            }
            
            prof.stop();
        }
}
