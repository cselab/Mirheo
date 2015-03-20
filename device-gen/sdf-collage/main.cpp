/*
 *  main.cpp
 *  Part of CTC/device-gen/sdf-collage/
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
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <assert.h>
#include <string>

using namespace std;

#define _ACCESS(f, x, y) f[(x) + xsize * (y)]

namespace Redistancing
{
    int xsize;
    float * phi0, * phi;
    float dt, invdx, invdy;

    template<int d>
    inline bool anycrossing_dir(int ix, int iy, const float sgn0) 
    {
        const int dx = d == 0, dy = d == 1, dz = d == 2;
        
        const float fm1 = _ACCESS(phi0, ix - dx, iy - dy);
        const float fp1 = _ACCESS(phi0, ix + dx, iy + dy);
        
        return (fm1 * sgn0 < 0 || fp1 * sgn0 < 0);
    }

    inline bool anycrossing(int ix, int iy, const float sgn0) 
    {
        return
        anycrossing_dir<0>(ix, iy, sgn0) ||
        anycrossing_dir<1>(ix, iy, sgn0) ;
    }
    
    float sussman_scheme(int ix, int iy, float sgn0)
    {
        const float phicenter =  _ACCESS(phi, ix, iy);
        
        const float dphidxm = phicenter -     _ACCESS(phi, ix - 1, iy);
        const float dphidxp = _ACCESS(phi, ix + 1, iy) - phicenter;
        const float dphidym = phicenter -     _ACCESS(phi, ix, iy - 1);
        const float dphidyp = _ACCESS(phi, ix, iy + 1) - phicenter;
        
        if (sgn0 == 1)
        {
            const float xgrad0 = max( max((float)0, dphidxm), -min((float)0, dphidxp)) * invdx;
            const float ygrad0 = max( max((float)0, dphidym), -min((float)0, dphidyp)) * invdy;
            
            const float G0 = sqrtf(xgrad0 * xgrad0 + ygrad0 * ygrad0) - 1;
            
            return phicenter - dt * sgn0 * G0;
        }
        else
        {
            const float xgrad1 = max( -min((float)0, dphidxm), max((float)0, dphidxp)) * invdx;
            const float ygrad1 = max( -min((float)0, dphidym), max((float)0, dphidyp)) * invdy;
            
            const float G1 = sqrtf(xgrad1 * xgrad1 + ygrad1 * ygrad1) - 1;
            
            return phicenter - dt * sgn0 * G1;
        }
    }
        
    void redistancing(const int iterations, const float dt, const float dx, const float dy,
                      const int xsize, const int ysize,
                      float * field)
    {
        Redistancing::xsize = xsize;
        Redistancing::dt = dt;
        Redistancing::invdx = 1. / dx;
        Redistancing::invdy = 1. / dy;
        
        Redistancing::phi0 = new float[xsize * ysize];
        memcpy(phi0, field, sizeof(float) * xsize * ysize);
        Redistancing::phi = field;
        
        float * tmp = new float[xsize * ysize];
        for(int t = 0; t < iterations; ++t)
        {
            if (t % 30 == 0)
                    printf("t: %d\n", t);
            
#pragma omp parallel for
            for(int iy = 0; iy < ysize; ++iy)
                for(int ix = 0; ix < xsize; ++ix)
                {
                    const float myval0 = _ACCESS(phi0, ix, iy);
                    const float sgn0 = myval0 > 0 ? 1 : (myval0 < 0 ? -1 : 0);
                    
                    if (anycrossing(ix, iy, sgn0) || ix == 0 || ix == xsize - 1 || iy == 0 || iy == ysize - 1)
                        tmp[ix + xsize * iy] = myval0;
                    else
                        tmp[ix + xsize * iy] = sussman_scheme(ix, iy, sgn0);
                }
            
            memcpy(field, tmp, sizeof(float) * xsize * ysize);
        }
        
        delete [] tmp;
        delete [] phi0;
        phi0 = NULL;
    }
}

void mergeSDF(int NX, int NY, vector< vector<float> >& cookie, vector<float>& cake)
{
    cake.resize(cookie.size() * cookie[0].size());
    printf("SIZE: %d\n", cake.size());
    const int stride = NX;
    for(int iy = 0; iy < cookie.size() * NY; ++iy)
        for(int ix = 0; ix < NX; ++ix)
        {
            const int dst = ix + stride * iy;
            const int iobst = iy / NY;
            cake[dst] = cookie[iobst][ix + NX * (iy - iobst*NY)];
        }
}

int main(int argc, char ** argv)
{
    if (argc != 5)
    {
        printf("usage: ./sdf-collage <input-2d-sdf> <xtimes> <ytimes> <output-2d-sdf>\n");
        return -1;
    }
    
    const int xtimes = atoi(argv[2]);
    int ytimes = atoi(argv[3]);
    
    float xextent, yextent, zextent;
    int NX, NY,NZ;
    vector< vector<float> > cookie;
    vector<float>  cake;
    if (string(argv[1]) != "files.txt")
    {
        cookie.resize(1);
        // for one file
        FILE * f = fopen(argv[1], "r");
        assert(f != 0);
        fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextent);
        fscanf(f, "%d %d %d\n", &NX, &NY, &NZ);
        printf("Extent: [%f, %f, %f]. Grid size: [%d, %d, %d]\n", xextent, yextent, zextent, NX, NY,NZ);
        assert(NZ == 1);
        cookie[0].resize(NX * NY * NZ, 0.0f);
        fread(&cookie[0][0], sizeof(float), NX * NY * NZ, f);
        fclose(f);
        
        printf("Populate %d * %d times\n", xtimes, ytimes);
        const int stride = xtimes * NX;
        cake.resize(xtimes * ytimes * NX * NY);

        for(int ty = 0; ty < ytimes; ++ty)
        for(int tx = 0; tx < xtimes; ++tx) {
            for(int iy = 0; iy < NY; ++iy)
                for(int ix = 0; ix < NX; ++ix)
                {
                    const int gx = ix + NX * tx;
                    const int gy = iy + NY * ty;
                    const int dst = gx + stride * gy;
                    
                    assert(dst < cake.size());
                    assert(ix + NX * iy < cookie[0].size());
                    cake[dst] = cookie[0][ix + NX * iy];
                }
        }
    } else {
        vector<string> files;
        
        FILE* fs = fopen(argv[1], "r");
        assert(fs != 0);
        string buf(127, ' ');
        while(fscanf(fs, "%s\n", &buf[0]) == 1) {
            files.push_back(buf);
        }
        fclose(fs);
        
        ytimes = files.size();
        cookie.resize(ytimes);
        assert(xtimes == 1);

        for (int i = files.size() - 1; i >= 0; --i)
        {
            printf("Reading file %s ...\n", files[i].c_str());
            FILE * f = fopen(files[i].c_str(), "r");
            assert(f != 0);
            fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextent);
            fscanf(f, "%d %d %d\n", &NX, &NY, &NZ);
            printf("Extent: [%g, %g, %g]. Grid size: [%d, %d, %d]\n", xextent, yextent, zextent, NX, NY,NZ);
            assert(NZ == 1);
            cookie[i].resize(NX * NY * NZ, 0.0f);
            fread(&cookie[i][0], sizeof(float), NX * NY * NZ, f);
            fclose(f);
        }

        mergeSDF(NX, NY, cookie, cake); 
    }
    
    const float dx = xextent / NX;
    const float dy = yextent / NY;
    Redistancing::redistancing(240, 0.25 * min(dx, dy), dx, dy, xtimes * NX, ytimes * NY, &cake[0]);
    
    {
        FILE * f = fopen(argv[4], "w");
        assert(f != 0);
        fprintf(f, "%f %f %f\n", xtimes * xextent, ytimes * yextent, 1.0f);
        fprintf(f, "%d %d %d\n", xtimes * NX, ytimes * NY, 1);
        fwrite(&cake[0], sizeof(float), cake.size(), f);
        fclose(f);
    }
    
    return 0;
}
