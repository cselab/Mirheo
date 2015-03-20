/*
 *  cell-factory.h
 *  Part of CTC/cuda-dpd-sem/sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-08-08.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

struct ParamsSEM
{
	float rcutoff, gamma, u0, rho, req, D, rc;
};

void cell_factory(int n, float * xyz, ParamsSEM& params);

