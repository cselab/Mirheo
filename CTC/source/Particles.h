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
	real *x,  *y,  *z;
	real *vx, *vy, *vz;
	real *ax, *ay, *az;
	real *bx, *by, *bz; // olala
	real *m;
	real *tmp;
    
    int *label;
	int n;
};
