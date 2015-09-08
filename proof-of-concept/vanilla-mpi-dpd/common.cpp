/*
 *  common.cpp
 *  Part of uDeviceX/vanilla-mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-07.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include "common.h"

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;
