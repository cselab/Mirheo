#include "redistance.h"

namespace Redistancing
{
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
        Redistancing::ysize = ysize;
        Redistancing::dt = dt;
        Redistancing::invdx = 1. / dx;
        Redistancing::invdy = 1. / dy;

        for(int code = 0; code < 3 * 3; ++code)
        {
            if (code == 1 + 3) continue;

            const float deltax = dx * ((code % 3) - 1);
            const float deltay = dy * ((code % 9) / 3 - 1);

            const float dl = sqrtf(deltax * deltax + deltay * deltay);

            Redistancing::dls[code] = dl;
        }


        Redistancing::phi0 = new float[xsize * ysize];
        memcpy(phi0, field, sizeof(float) * xsize * ysize);
        Redistancing::phi = field;

        float * tmp = new float[xsize * ysize];
        for(int t = 0; t < iterations; ++t)
        {
            if (t % 100 == 0)
                printf("t: %d, size: %d %d\n", t, xsize, ysize);

//#pragma omp parallel for
            for(int iy = 0; iy < ysize; ++iy)
                for(int ix = 0; ix < xsize; ++ix)
                {
                    const float myval0 = _ACCESS(phi0, ix, iy);
                    const float sgn0 = myval0 > 0 ? 1 : (myval0 < 0 ? -1 : 0);

                    const bool boundary = (
                        ix == 0 || ix == xsize - 1 ||
                        iy == 0 || iy == ysize - 1);
                    if (boundary)
                        tmp[ix + xsize * iy] = simple_scheme(ix, iy, sgn0, myval0);
                    else
                    {
                        if (anycrossing(ix, iy, sgn0))
                            tmp[ix + xsize * iy] = myval0;
                        else
                            tmp[ix + xsize * iy] = sussman_scheme(ix, iy, sgn0);
                    }
                    assert(fabs(tmp[ix + xsize * iy]) < 1e7);
                }

            memcpy(field, tmp, sizeof(float) * xsize * ysize);
        }

        delete [] tmp;
        delete [] phi0;
        phi0 = NULL;
    }

    float simple_scheme(int ix, int iy, float sgn0, float myphi0)
    {
        float mindistance = 1e6f;
        for(int code = 0; code < 3 * 3; ++code)
        {
            if (code == 1 + 3) continue;

            const int xneighbor = ix + (code % 3) - 1;
            const int yneighbor = iy + (code % 9) / 3 - 1;

            if (xneighbor < 0 || xneighbor >= xsize) continue;
            if (yneighbor < 0 || yneighbor >= ysize) continue;

            const float phi0_neighbor = _ACCESS(phi0, xneighbor, yneighbor);
            const float phi_neighbor = _ACCESS(phi, xneighbor, yneighbor);

            const float dl = Redistancing::dls[code];

            float distance = 0;

            if (sgn0 * phi0_neighbor < 0)
                distance = - myphi0 * dl / (phi0_neighbor - myphi0);
            else
                distance = dl + abs(phi_neighbor);

            mindistance = min(mindistance, distance);
        }

        return sgn0 * mindistance;
    }

}
