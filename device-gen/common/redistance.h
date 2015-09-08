/*
 *  resistance.h
 *  Part of CTC/device-gen/common/
 *
 *  Created and authored by Diego Rossinelli and Kirill Lykov on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cmath>
#include <cstring>

using namespace std;

#define _ACCESS(f, x, y) f[(x) + xsize * (y)]

namespace Redistancing
{
    int xsize, ysize;
    float * phi0, * phi;
    float dt, invdx, invdy;
    float dls[9];

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
    
    float simple_scheme(int ix, int iy, float sgn0, float myphi0);
    
    float sussman_scheme(int ix, int iy, float sgn0);
        
    void redistancing(const int iterations, const float dt, const float dx, const float dy,
                      const int xsize, const int ysize, float * field);
}
