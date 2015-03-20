/*
 *  redistancing.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-17.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

void redistancing(float * host_inout, const int NX, const int NY, const int NZ, const float dx, const float dy, const float dz,
			 const int niterations);
