//
//  Particles.h
//  CTC
//
//  Created by Dmitry Alexeev on 09.07.14.
//  Copyright (c) 2014 Dmitry Alexeev. All rights reserved.
//

#pragma once

#include "Misc.h"

//**********************************************************************************************************************
// Particles
//**********************************************************************************************************************
struct Particles
{
    real *xdata, *vdata, *adata;
    
    inline real& x (const int i) {return xdata[3*i + 0];}
    inline real& y (const int i) {return xdata[3*i + 1];}
    inline real& z (const int i) {return xdata[3*i + 2];}
    inline real& vx(const int i) {return xdata[3*i + 0];}
    inline real& vy(const int i) {return xdata[3*i + 1];}
    inline real& vz(const int i) {return xdata[3*i + 2];}
    inline real& ax(const int i) {return xdata[3*i + 0];}
    inline real& ay(const int i) {return xdata[3*i + 1];}
    inline real& az(const int i) {return xdata[3*i + 2];}
    
    real *m;
	real *tmp;
    
    int *label;
	int n;
};
