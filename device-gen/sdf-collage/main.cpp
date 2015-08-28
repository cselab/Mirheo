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
#include <limits>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <assert.h>
#include <string>
#include <iostream>
#include "../common/redistance.h"
#include "../common/common.h"
using namespace std;

void mergeSDF(int NX, int NY, const vector< vector<float> >& cookie, vector<float>& cake)
{
    cake.resize(cookie.size() * cookie[0].size());
    printf("SIZE: %d\n", cake.size());
    const int stride = NX;
    for(int iy = 0; iy < cookie.size() * NY; ++iy)
        for(int ix = 0; ix < NX; ++ix)
        {
            const int dst = ix + stride * iy;
            const int iobst = iy / NY;
            assert(ix + NX * (iy - iobst*NY) < cookie[iobst].size());
            cake[dst] = cookie[iobst][ix + NX * (iy - iobst*NY)];
        }
}

int main(int argc, char ** argv)
{
    if (argc != 6)
    {
        printf("usage: ./sdf-collage <input-2d-sdf> <xtimes> <ytimes> <ymargin>  <output-2d-sdf>\n");
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
        readDAT(argv[1], cookie[0], xextent, yextent, zextent, NX, NY, NZ);
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
            readDAT(files[i].c_str(), cookie[i], xextent, yextent, zextent, NX, NY,NZ);
            for(int iy = 0; iy < NY; ++iy)
                for(int ix = 0; ix < NX; ++ix)
                {
                    if (cookie[i][ix + NX * iy] > 1e3)
                    {
                        std::cout << "ERROR in file " << files[i].c_str() << std::endl;
                        exit(0);
                    }
                }
        }

        mergeSDF(NX, NY, cookie, cake);

        // add walls in Y directio
        float wallWidth = atof(argv[4]);
        if (wallWidth != 0.0f)
        {
            int cakeNX = xtimes * NX;
            int cakeNY = ytimes * NY;
            const float x0 = -xtimes * xextent * 0.5;
            const float dx = xtimes * xextent / (cakeNX - 1);

            const float y0 = -ytimes * yextent * 0.5;
            const float dy = ytimes * yextent / (cakeNY - 1);

            const float angle = (1.8/180.)*M_PI;
            const float normal[] = {-cos(angle), sin(angle)};
            wallWidth = -2*y0*tan(angle);
            
            float ypick = 25.0f; //15
            float widthOfBufferZone = 8-wallWidth +  0.0*(48 - 2*wallWidth);
            float xpick = (wallWidth) * (-2.0f*y0 - ypick) / (-2.0f*y0);
            const float angle2 = atan(xpick/ypick);
            std::cout << "YY = " << xpick << ", " << y0 + ypick << " ANGLE = "  << angle2/M_PI*180  << std::endl;
            const float normal2[] = {-cos(angle2), -sin(angle2)};
            
            
            const float linePoint[] = {-x0 - wallWidth, y0};
            const float linePoint2[] = {-x0 - xpick -wallWidth, -y0 - ypick};

            for(int iy = 0; iy < cakeNY; ++iy)
                for(int ix = 0; ix < cakeNX; ++ix)
                {
                    const float signX = sign(dx*ix + x0);
                    float p[] = {dx*ix + x0, dy*iy + y0};
                    //float xsdf = std::numeric_limits<float>::min();
                        float padding = signbit(-p[0])*widthOfBufferZone;
                        float xsdf = -1e6;
                        
                        if ((signX == -1 && p[1] > (y0 + ypick)) || (signX == 1 && p[1] < (-y0 - ypick))) {
                            xsdf = -(normal[0]*(fabs(p[0]) - linePoint[0] + padding) - signX*normal[1]*(p[1] - linePoint[1]));
                        } else { 
                            xsdf = -(normal2[0]*(fabs(p[0]) - linePoint2[0] - signbit(p[0])*wallWidth + padding) - normal2[1]*(fabs(p[1]) - linePoint2[1]));   
                        }
                                    
                            cake[ix + cakeNX*iy] = std::max(cake[ix + cakeNX*iy], xsdf);
                    //}                    
                    //const float xsdf = fabs(dx*ix + x0) - (xextent * 0.5 - wallWidth);
                    //cake[ix + cakeNX*iy] = std::max(cake[ix + cakeNX*iy], xsdf);
                }
        }         
    }
    
    const float dx = xextent / (NX - 1);
    const float dy = yextent / (NY - 1);
    Redistancing::redistancing(1000, 0.25 * min(dx, dy), dx, dy, xtimes * NX, ytimes * NY, &cake[0]);
   
    writeDAT(argv[5], cake, xtimes * xextent, ytimes * yextent, 1.0f, xtimes * NX, ytimes * NY, 1);
    return 0;
}

