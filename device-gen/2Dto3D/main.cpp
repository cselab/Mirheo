/*
 *  main.cpp
 *  Part of CTC/device-gen/2Dto3D/
 *
 *  Created and authored by Diego Rossinelli and Kirill Lykov on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <assert.h>
#include "../common/common.h"

using namespace std;

int main(int argc, char ** argv)
{
    if (argc != 6)
    {
        printf("usage: ./2to3 <input-filename> <zextent> <zmargin> <NZ> <output-filename>\n");
        return 1;
    }
    
    const float zextent = atof(argv[2]);
    const float zmargin = atof(argv[3]);
    const int NZ = atoi(argv[4]);

    int NX, NY;
    float xextent, yextent;    

    vector<float> slice;    
    int oldNZ;
    float zextentOld;
    readDAT(argv[1], slice, xextent, yextent, zextentOld, NX, NY, oldNZ);
    assert(oldNZ == 1);
        
    printf("Generating data with extent [%f, %f, %f], dimensions [%d, %d, %d], zmargin %f\n", 
        xextent, yextent, zextent + 2 * zmargin, NX, NY, NZ, zmargin);
    
    vector<float>  outputslice(NX * NY, 0.0f);
    
    const float z0 = -zextent * 0.5 - zmargin;
    const float dz = (zextent + 2 * zmargin) / (NZ - 1);
    
    FILE * f = fopen(argv[5], "w");
    assert(f != 0);
    fprintf(f, "%f %f %f\n", yextent, xextent, zextent + 2.0f * zmargin);
    fprintf(f, "%d %d %d\n", NY, NX, NZ);

    for(int iz = 0; iz < NZ; ++iz)
    {
        const float z = z0 + iz * dz;
        
#pragma omp parallel for
        for(int iy = 0; iy < NY; ++iy)
            for(int ix = 0; ix < NX; ++ix)
            {
                const float xysdf = slice[ix + NX * (NY - 1 - iy)]; // NY -1 to change Y-axis direction
                float val = xysdf;
                if (zmargin != 0.0f) {
                    const float zsdf = fabs(z) - zextent * 0.5;
                    if (xysdf < 0)
                        val = max(zsdf, xysdf);
                    else
                        val = (zsdf < 0) ? xysdf : sqrt(zsdf * zsdf + xysdf * xysdf);
                }
                assert(iy + NY * (ix) < outputslice.size());
               
                assert(fabs(val) < 1e3); // to check that the value has reasonable range
                outputslice[iy + NY * ix] = val;
            }
        
        if (iz == 0)
        {
            unsigned char * ptr = (unsigned char *)&outputslice[0];
            if ((ptr[0] >= 9 && ptr[0] <= 13) || ptr[0] == 32 )
            {
                ptr[0] = (ptr[0] == 32) ? 33 : (ptr[0] < 11) ? 8 : 14;
                printf("INFO: some symbols were changed while writing\n");
            }
        }
        
        int result = fwrite(&outputslice.front(), sizeof(float), NX * NY, f);
        
        if (result != NX  * NY) {
            printf("ERROR: written less than expected");
            exit(3);
        }

    }
   
    fclose(f);
}

