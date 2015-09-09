/*
 *  main.cpp
 *  Part of CTC/device-gen/sdf-unit-par/
 *
 *  Created and authored by Kirill Lykov on 2015-08-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include "../common/common.h"
using namespace std;

#define REAL float

REAL distToSide(REAL x, REAL y, REAL radius) {
    return sqrt(x*x + y*y) - radius;
}

int main(int argc, char ** argv)
{
    if (argc != 4)
    {
        printf("usage: ./sdf-cylinder <NX> <radius> <out-file-name> \n");
        return 1;
    }

    const int N = atoi(argv[1]);
    const REAL radius = atof(argv[2]);;
    const REAL extent = 2.0f*radius + 4.0f;

    std:cout << "Will generate SDF with extent " << extent << " " << extent 
             << ". Grid size " << N << " x " << N << std::endl;

    const REAL xlb = -extent/2.0f;
    const REAL ylb = -extent/2.0f; 

    vector<REAL> sdf(N * N, 0.0f);
    const REAL dx = extent / (N-1);
    const REAL dy = extent / (N-1);

    for(int iy = 0; iy < N; ++iy)
    for(int ix = 0; ix < N; ++ix)
    {
        const REAL x = xlb + ix * dx;
        const REAL y = ylb + iy * dy;
        
        sdf[ix + N * iy] =  distToSide(x, y, radius);
    }
    
    writeDAT(argv[3], sdf, extent, extent, REAL(1.0), N, N, 1);
    
    return 0;
}
