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
    
    {   
        printf("Reading file %s...\n", argv[1]);
        FILE * f = fopen(argv[1], "r");
        assert(f != 0);
        float zextentOld;
        int NZOld;
        fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextentOld);
        fscanf(f, "%d %d %d\n", &NX, &NY, &NZOld);
        printf("Extent: [%f, %f, %f]. Grid size: [%d, %d, %d]\n", xextent, yextent, zextentOld, NX, NY,NZOld);
        slice.resize(NX * NY, 0.0f);
        fread(&slice[0], sizeof(float), slice.size(), f);
        fclose(f);
    }
        
    printf("Generating data with extent [%f, %f, %f], dimensions [%d, %d, %d], zmargin %f\n", 
        xextent, yextent, zextent + 2 * zmargin, NX, NY, NZ, zmargin);
    vector<float>  volume(NX * NY * NZ, 0.0f);
    
    const float z0 = -zextent * 0.5 - zmargin;
    const float dz = (zextent + 2 * zmargin) / (NZ - 1);
    
//#pragma omp parallel for
    for(int iz = 0; iz < NZ; ++iz)
    {
        const float z = z0 + iz * dz;
        for(int iy = 0; iy < NY; ++iy)
            for(int ix = 0; ix < NX; ++ix)
            {
                const float xysdf = slice[ix + NX * (NY - 1 - iy)]; // NY -1 to change Y-axis direction
                const float zsdf = fabs(z) - zextent * 0.5;
                float val;
                if (xysdf < 0)
                    val = max(zsdf, xysdf);
                else
                    val = (zsdf < 0) ? xysdf : sqrt(zsdf * zsdf + xysdf * xysdf);
            
                assert(iy + NY * (ix + NX * iz) < volume.size());
                assert(volume[iy + NY * (ix + NX * iz)] == 0.0f);
                volume[iy + NY * (ix + NX * iz)] = val;
            }
    }
   
    {
        FILE * f = fopen(argv[5], "w");
        assert(f != 0);
        fprintf(f, "%f %f %f\n", yextent, xextent, zextent + 2.0f * zmargin); //exchange X and Y
        fprintf(f, "%d %d %d\n", NY, NX, NZ);
        fwrite(&volume[0], sizeof(float), volume.size(), f);
        fclose(f);
    }
}

