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

int main(int argc, char ** argv)
{
    if (argc != 6)
    {
        printf("usage: ./sdf-unit-egg <NX> <NY> <xextent> <yextent> <out-file-name> \n");
        return 1;
    }
    
    const int NX = atoi(argv[1]);
    const int NY = atoi(argv[2]);
    const float xextent = atof(argv[3]);
    const float yextent = atof(argv[4]);
 
    vector<float> xs, ys;
    Egg egg;
    egg.run(xs, ys);

    const float xlb = -xextent/2.0f;
    const float ylb = -yextent/2.0f; 
    printf("starting brute force sdf with %d x %d starting from %f %f to %f %f\n",
           NX, NY, xlb, ylb, xlb + xextent, ylb + yextent);
    
    vector<float> sdf(NX * NY, 0.0f);
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
    
    writeDAT(argv[5], sdf, xextent, yextent, 1.0f, NX, NY, 1);
    //FILE * f = fopen(argv[5], "w");
    //fprintf(f, "%f %f %f\n", xextent, yextent, 1.0f);
    //fprintf(f, "%d %d %d\n", NX, NY, 1);
    //fwrite(sdf, sizeof(float), NX * NY, f);
    //fclose(f);
    
    //delete [] sdf;
    
    return 0;
}

