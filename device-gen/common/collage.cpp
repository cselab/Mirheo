/*
 *  collage.cpp
 *  Part of CTC/device-gen/sdf-collage/
 *
 *  Created and authored by Diego Rossinelli and Kirill Lykov on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */
#include "collage.h"
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
#include "common.h"
using namespace std;

static void mergeSDF(int NX, int NY, const vector< vector<float> >& sampleSDF, vector<float>& outputSDF)
{
    outputSDF.resize(sampleSDF.size() * sampleSDF[0].size());
    printf("SIZE: %d\n", outputSDF.size());
    const int stride = NX;
    for(int iy = 0; iy < sampleSDF.size() * NY; ++iy)
        for(int ix = 0; ix < NX; ++ix)
        {
            const int dst = ix + stride * iy;
            const int iobst = iy / NY;
            assert(ix + NX * (iy - iobst*NY) < sampleSDF[iobst].size());
            outputSDF[dst] = sampleSDF[iobst][ix + NX * (iy - iobst*NY)];
        }
}

void populateSDF(const int NX, const int NY, const float xextent, const float yextent, const std::vector<float>& sampleSDF,
                 const int xtimes, const int ytimes, std::vector<float>& outputSDF)
{
    printf("Populate %d * %d times\n", xtimes, ytimes);
    const int stride = xtimes * NX;
    outputSDF.resize(xtimes * ytimes * NX * NY);

    for(int ty = 0; ty < ytimes; ++ty)
        for(int tx = 0; tx < xtimes; ++tx) {
            for(int iy = 0; iy < NY; ++iy)
                for(int ix = 0; ix < NX; ++ix)
                {
                    const int gx = ix + NX * tx;
                    const int gy = iy + NY * ty;
                    const int dst = gx + stride * gy;

                    assert(dst < outputSDF.size());
                    assert(ix + NX * iy < sampleSDF.size());
                    outputSDF[dst] = sampleSDF[ix + NX * iy];
                }
        }
}

void collageSDF(const int NX, const int NY, const float xextent, const float yextent, 
                const vector< vector<float> >& sampleSDF, const int ytimes,
                bool wallInY, vector<float>& outputSDF)
{        
    mergeSDF(NX, NY, sampleSDF, outputSDF);
    const int xtimes = 1;
    if (wallInY)
    {
        int outputSDFNX = xtimes * NX;
        int outputSDFNY = ytimes * NY;
        const float x0 = -xtimes * xextent * 0.5;
        const float dx = xtimes * xextent / (outputSDFNX - 1);

        const float y0 = -ytimes * yextent * 0.5;
        const float dy = ytimes * yextent / (outputSDFNY - 1);

        const float angle = (1.8/180.)*M_PI;
        const float normal[] = {-cos(angle), sin(angle)};
        const float wallWidth = -2*y0*tan(angle);
            
        float ypick = 25.0f; //15
        float widthOfBufferZone = 8-wallWidth +  0.0*(48 - 2*wallWidth);
        float xpick = (wallWidth) * (-2.0f*y0 - ypick) / (-2.0f*y0);
        const float angle2 = atan(xpick/ypick);
        std::cout << "YY = " << xpick << ", " << y0 + ypick << " ANGLE = "  << angle2/M_PI*180  << std::endl;
        const float normal2[] = {-cos(angle2), -sin(angle2)};
            
            
        const float linePoint[] = {-x0 - wallWidth, y0};
        const float linePoint2[] = {-x0 - xpick -wallWidth, -y0 - ypick};

        for(int iy = 0; iy < outputSDFNY; ++iy)
            for(int ix = 0; ix < outputSDFNX; ++ix)
            {
                const float signX = sign(dx*ix + x0);
                float p[] = {dx*ix + x0, dy*iy + y0};
                float padding = signbit(-p[0])*widthOfBufferZone;
                float xsdf = -1e6;
                      
                if ((signX == -1 && p[1] > (y0 + ypick)) || (signX == 1 && p[1] < (-y0 - ypick))) {
                    xsdf = -(normal[0]*(fabs(p[0]) - linePoint[0] + padding) - signX*normal[1]*(p[1] - linePoint[1]));
                } else { 
                    xsdf = -(normal2[0]*(fabs(p[0]) - linePoint2[0] - signbit(p[0])*wallWidth + padding) - normal2[1]*(fabs(p[1]) - linePoint2[1]));   
                }
                                    
                outputSDF[ix + outputSDFNX*iy] = std::max(outputSDF[ix + outputSDFNX*iy], xsdf);
            }
    }         
}

void shiftSDF(const int NX, const int NY, const float xextent, const float yextent, const std::vector<float>& inputGrid, 
              const float xshift, const float xpadding, int& newNX, float& newXextent, std::vector<float>& outGrid)
{
    float h = xextent / (NX - 1);
    int ixshift = xshift / h;
    int ipadding = xpadding / h + 1;
    newNX = NX + ipadding;
    newXextent = xextent + xpadding;

    float minVal = *std::min(inputGrid.begin(), inputGrid.end());
    outGrid.resize(newNX * NY, -1e6);

    for(int iy = 0; iy < NY; ++iy)
        for(int ix = 0; ix < NX ; ++ix)
        {
            int newIx = (ix + ixshift) % newNX;
            assert(fabs(inputGrid[ix + NX * iy]) < 1e3);
            outGrid[newIx + newNX * iy] = inputGrid[ix + NX * iy];
        }
}

