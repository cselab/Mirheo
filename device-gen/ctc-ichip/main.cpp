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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include "../common/common.h"
#include "../common/collage.h"
#include "../common/redistance.h"
#include "../common/2Dto3D.h"

using namespace std;

struct Egg 
{
    float r1, r2, alpha;

    Egg() 
    : r1(12.0f), r2(8.5f), alpha(0.03f) 
    {
    }    

    float x2y(float x) const {
        return sqrt(r2*r2 * exp(-alpha * x) * (1.0f - x*x/r1/r1));
    }
    
    void run(vector<float>& vx, vector<float>& vy) {
        int N = 500;
        float dx = 2.0f * r1 / (N - 1);
        for (int i = 0; i < N; ++i) {
            float x = i * dx - r1;
            float y = x2y(x);
            vx.push_back(x);
            vy.push_back(y);
        }
        
        auto vxRev = vx;
        vx.insert(vx.end(), vxRev.rbegin(), vxRev.rend());

        auto vyRev = vy;
        for_each(vyRev.begin(), vyRev.end(), [](float& i) { i *= -1.0f; });
        vy.insert(vy.end(), vyRev.rbegin(), vyRev.rend());
    }
};

void generateEggSDF(const int NX, const int NY, const float xextent, const float yextent, vector<float>& sdf)
{
    vector<float> xs, ys;
    Egg egg;
    egg.run(xs, ys);

    const float xlb = -xextent/2.0f;
    const float ylb = -yextent/2.0f;
    printf("starting brute force sdf with %d x %d starting from %f %f to %f %f\n",
           NX, NY, xlb, ylb, xlb + xextent, ylb + yextent);

    sdf.resize(NX * NY, 0.0f);
    const float dx = xextent / NX;
    const float dy = yextent / NY;
    const int nsamples = xs.size();

    for(int iy = 0; iy < NY; ++iy)
    for(int ix = 0; ix < NX; ++ix)
    {
        const float x = xlb + ix * dx;
        const float y = ylb + iy * dy;

        float distance2 = 1e6;
        int iclosest = 0;
        for(int i = 0; i < nsamples ; ++i)
        {
            const float xd = xs[i] - x;
            const float yd = ys[i] - y;
            const float candidate = xd * xd + yd * yd;

            if (candidate < distance2)
            {
                iclosest = i;
                distance2 = candidate;
            }
        }

        float s = -1;

        {
            const float ycurve = egg.x2y(x);
            if (x >= -egg.r1 && x <= egg.r1 && fabs(y) <= ycurve)
                s = +1;
        }


        sdf[ix + NX * iy] = s * sqrt(distance2);
    }
}

typedef vector<float> SDF;

int main(int argc, char ** argv)
{
    
    int nColumns = 5;
    int nRows = 57;
    int nrepeat = 2;

    int zmargin = 5.0f;

    // 1 Create 2D SDF for 1 obstacle
    const float eggSizeX = 56.0f; // size of the egg with the empty space aroung it
    const float eggSizeY = 32.0f;
    const float eggSizeZ = 58.0f;
    const float resolution = 1.0f; // how many grid point per one micron 
    const int eggNX = static_cast<int>(eggSizeX * resolution);
    const int eggNY = static_cast<int>(eggSizeY * resolution);
    const int eggNZ = static_cast<int>(eggSizeZ * resolution);
 
    SDF eggSdf;
    generateEggSDF(eggNX, eggNY, eggSizeX, eggSizeY, eggSdf);   
    writeDAT("out.dat", eggSdf, eggNX, eggNY, 1, eggSizeX, eggSizeY, 1.0f);
    
    // 2 Create 1 row of obstacles
    int rowNX = nColumns*eggNX;
    int rowNY = eggNY;
    int rowSizeX = nColumns*eggSizeX;
    int rowSizeY = eggSizeY;
    SDF rowObstacles;
    populateSDF(eggNX, eggNY, eggSizeX, eggSizeY, eggSdf, nColumns, 1, rowObstacles);

    // 3 Shift rows
    const float angle = 1.7f * M_PI / 180.0f;
    const int nRowsPerShift = static_cast<int>(ceil(eggSizeX / (eggSizeY * tan(angle))));
    if (fabs(eggSizeX / (eggSizeY * tan(angle)) - nRowsPerShift) > 1e-1) {
        std::cout << "ERROR: Suggest changing the angle\n";    
        return 1;
    }

    float padding = float(ceil(nRows * eggSizeY * tan(angle)));
    // TODO Do I need this nUniqueRows?
    int nUniqueRows = nRows;
    if (nRows > nRowsPerShift) {
        nUniqueRows = nRowsPerShift;
        padding = float(round(nRowsPerShift * eggSizeY * tan(angle)));
    }

    // TODO fix this stupid workaround
    if (padding < 32.0f)
        padding = 0.0f;
    if (padding == 57.0f)
        padding = 56.0f;
    padding = padding + 8; // adjust padding to have desired size
    
    std::cout << "Launching rows generation. Padding = "<< padding <<std::endl;
    std::vector<SDF> shiftedRows(nUniqueRows);
    int shiftedRowNX = 0; // they are all the same length
    float shiftedRowSizeX = 0.0f;
    //for (int i = nUniqueRows-1; i >= 0; --i) {
    for (int i = 0; i < nUniqueRows; ++i) {
        float xshift = (nUniqueRows - i -1 ) * 32.0f * tan(angle);
        shiftSDF(rowNX, rowNY, rowSizeX, rowSizeY, rowObstacles, xshift, padding, shiftedRowNX, shiftedRowSizeX, shiftedRows[i]);        
    }

    // 4 Collage rows
    SDF finalSDF;
    collageSDF(shiftedRowNX, rowNY, shiftedRowSizeX, rowSizeY, shiftedRows, nRows, true, finalSDF);

    // 5 Apply redistancing for the result
    float finalExtent[] = {shiftedRowSizeX, nRows*rowSizeY};
    int finalN[] = {shiftedRowNX, nRows*rowNY};
    const float dx = finalExtent[0] / (finalN[0] - 1);
    const float dy = finalExtent[1] / (finalN[1] - 1);
    Redistance redistancer(0.25f * min(dx, dy), dx, dy, finalN[0], finalN[1]);
    redistancer.run(1e2, &finalSDF[0]);

    // 6 Repeat this pattern
    SDF finalSDF2;
    populateSDF(finalN[0], finalN[1], finalExtent[0], finalExtent[1], finalSDF, 1, nrepeat, finalSDF2);    
    std::swap(finalSDF, finalSDF2);   
 
    // 6 Write result to the file
    writeDAT("2d.dat", finalSDF, finalN[0], nrepeat * finalN[1], 1, finalExtent[0], nrepeat*finalExtent[1], 1.0f);

    conver2Dto3D(finalN[0], nrepeat * finalN[1], finalExtent[0], nrepeat*finalExtent[1], finalSDF, 
                 eggNZ, eggSizeZ - 2.0f*zmargin, zmargin, "3d.dat");

    return 0;
}

