/*
 *  rbcvector-cpu-utils.h
 *  ctc phenix
 *
 *  Created by Dmitry Alexeev on Nov 19, 2014
 *  Copyright 2014 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>

#include "rbcvector.h"

using namespace std;

void cuda_loadHeader(RBCVector& rbcs, const char* fname, bool report = true);
void cuda_initUnique(RBCVector& rbcs, vector<vector<float> > origins, vector<float> coolo, vector<float> coohi);
void cuda_boundsAndComs(RBCVector& rbcs);
void cuda_eraseCell(RBCVector& rbcs, int id);  // TODO



