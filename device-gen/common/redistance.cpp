#include "redistance.h"
#include <algorithm>
#include <cassert>

    float Redistance::sussman_scheme(int ix, int iy, float sgn0)
    {
        const float phicenter =  _ACCESS(m_phi, ix, iy);

        const float dphidxm = phicenter -     _ACCESS(m_phi, ix - 1, iy);
        const float dphidxp = _ACCESS(m_phi, ix + 1, iy) - phicenter;
        const float dphidym = phicenter -     _ACCESS(m_phi, ix, iy - 1);
        const float dphidyp = _ACCESS(m_phi, ix, iy + 1) - phicenter;

        if (sgn0 == 1)
        {
            const float xgrad0 = std::max( max(0.0f, dphidxm), -std::min(0.0f, dphidxp)) * m_invdx;
            const float ygrad0 = std::max( max(0.0f, dphidym), -min(0.0f, dphidyp)) * m_invdy;

            const float G0 = sqrtf(xgrad0 * xgrad0 + ygrad0 * ygrad0) - 1.0f;

            return phicenter - m_dt * sgn0 * G0;
        }
        else
        {
            const float xgrad1 = std::max( -min(0.0f, dphidxm), std::max(0.0f, dphidxp)) * m_invdx;
            const float ygrad1 = std::max( -min(0.0f, dphidym), std::max(0.0f, dphidyp)) * m_invdy;

            const float G1 = sqrtf(xgrad1 * xgrad1 + ygrad1 * ygrad1) - 1.0f;

            return phicenter - m_dt * sgn0 * G1;
        }
    }

Redistance::Redistance(const float dt, const float dx, const float dy,
                      const int xsize, const int ysize)
: m_xsize(xsize), m_ysize(ysize), m_dt(dt), m_dx(dx), m_dy(dy), m_invdx(1.0f/dx), m_invdy(1.0f/dy), m_phi0(nullptr), m_phi(nullptr)
{}

    void Redistance::run(const int iterations, float * field)
    {
        for(int code = 0; code < 3 * 3; ++code)
        {
            if (code == 1 + 3) continue;

            const float deltax = m_dx * ((code % 3) - 1);
            const float deltay = m_dy * ((code % 9) / 3 - 1);

            const float dl = sqrtf(deltax * deltax + deltay * deltay);

            m_dls[code] = dl;
        }

        m_phi0 = new float[m_xsize * m_ysize];
        memcpy(m_phi0, field, sizeof(float) * m_xsize * m_ysize);
        m_phi = field;

        float * tmp = new float[m_xsize * m_ysize];
        for(int t = 0; t < iterations; ++t)
        {
            if (t % 100 == 0)
                printf("t: %d, size: %d %d\n", t, m_xsize, m_ysize);

//#pragma omp parallel for
            for(int iy = 0; iy < m_ysize; ++iy)
                for(int ix = 0; ix < m_xsize; ++ix)
                {
                    const float myval0 = _ACCESS(m_phi0, ix, iy);
                    const float sgn0 = myval0 > 0 ? 1 : (myval0 < 0 ? -1 : 0);

                    const bool boundary = (
                        ix == 0 || ix == m_xsize - 1 ||
                        iy == 0 || iy == m_ysize - 1);
                    if (boundary)
                        tmp[ix + m_xsize * iy] = simple_scheme(ix, iy, sgn0, myval0);
                    else
                    {
                        if (anycrossing(ix, iy, sgn0))
                            tmp[ix + m_xsize * iy] = myval0;
                        else
                            tmp[ix + m_xsize * iy] = sussman_scheme(ix, iy, sgn0);
                    }
                    assert(fabs(tmp[ix + m_xsize * iy]) < 1e7);
                }

            memcpy(field, tmp, sizeof(float) * m_xsize * m_ysize);
        }

        delete [] tmp;
        delete [] m_phi0;
        m_phi0 = nullptr;
    }

    float Redistance::simple_scheme(int ix, int iy, float sgn0, float myphi0)
    {
        float mindistance = 1e6f;
        for(int code = 0; code < 3 * 3; ++code)
        {
            if (code == 1 + 3) continue;

            const int xneighbor = ix + (code % 3) - 1;
            const int yneighbor = iy + (code % 9) / 3 - 1;

            if (xneighbor < 0 || xneighbor >= m_xsize) continue;
            if (yneighbor < 0 || yneighbor >= m_ysize) continue;

            const float phi0_neighbor = _ACCESS(m_phi0, xneighbor, yneighbor);
            const float phi_neighbor = _ACCESS(m_phi, xneighbor, yneighbor);

            const float dl = m_dls[code];

            float distance = 0;

            if (sgn0 * phi0_neighbor < 0)
                distance = - myphi0 * dl / (phi0_neighbor - myphi0);
            else
                distance = dl + abs(phi_neighbor);

            mindistance = std::min(mindistance, distance);
        }

        return sgn0 * mindistance;
    }

