/*
 *  main.cpp
 *  Part of CTC/device-gen/sdf-unit-par/
 *
 *  Created and authored by Kirill Lykov on 2015-03-28.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include "../common/common.h"
#include "../common/redistance.h"
#include <iostream>
#include <algorithm>

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        printf("usage: ./sdf-shift <input-file> <xshift> <xpadding> <output-file>\n");
        return -1;
    }
    
    float xshift = atof(argv[2]);
    float xpadding = atof(argv[3]);
    
    float xextent, yextent, zextent;
    int NX, NY,NZ;
    std::vector<float> inputGrid;
    readDAT(argv[1], inputGrid, xextent, yextent, zextent, NX, NY, NZ);    

    float h = xextent / (NX - 1);
    int ixshift = xshift / h;
    int ipadding = xpadding / h + 1;

    float minVal = *std::min(inputGrid.begin(), inputGrid.end());
    std::vector<float> outGrid((NX + ipadding) * NY * NZ, -1e6);
    assert(NZ == 1);
    
    for(int iy = 0; iy < NY; ++iy)
        for(int ix = 0; ix < NX ; ++ix)
        {   
            int newIx = (ix + ixshift) % (NX + ipadding);
            assert(fabs(inputGrid[ix + NX * iy]) < 1e3);
            outGrid[newIx + (NX + ipadding) * iy] = inputGrid[ix + NX * iy];
        }

    const float dx = (xextent + xpadding)/ (NX + ipadding - 1);
    const float dy = yextent / (NY - 1);
    Redistancing::redistancing(1000, 0.25 * min(dx, dy), dx, dy, NX + ipadding, NY, &outGrid[0]);

    writeDAT(argv[4], outGrid, xextent + xpadding, yextent, zextent, NX + ipadding, NY, NZ);
    
    
    return 0;
}
