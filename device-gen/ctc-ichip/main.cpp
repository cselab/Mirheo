/*
 *  main.cpp
 *  Part of CTC/device-gen/ctc-ichip/
 *
 *  Created and authored by Kirill Lykov on 2015-09-7.
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
#include <stdexcept>
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

typedef vector<float> SDF;

class CTCiChip1Builder
{
    int m_ncolumns, m_nrows, m_nrepeat;
    float m_resolution, m_zmargin;
    float m_eggSizeX, m_eggSizeY, m_eggSizeZ; // size of the egg with the empty space aroung it
    const float m_angle;
    std::string m_outFileName2D, m_outFileName3D; 
public:
    CTCiChip1Builder()
    : m_ncolumns(0), m_nrows(0), m_nrepeat(0), m_resolution(0), m_zmargin(0),
      m_eggSizeX(56.0f), m_eggSizeY(32.0f), m_eggSizeZ(58.0f),
      m_angle(1.7f * M_PI / 180.0f)
    {}

    CTCiChip1Builder& setNColumns(int ncolumns) 
    {
        m_ncolumns = ncolumns;
        return *this;
    }

    CTCiChip1Builder& setNRows(int nrows) 
    {
        m_nrows = nrows;
        return *this;
    }

    CTCiChip1Builder& setRepeat(float nrepeat)
    {
        m_nrepeat = nrepeat;
        return *this;
    }

    CTCiChip1Builder& setResolution(float resolution)          
    {
        m_resolution = resolution;
        return *this;
    }

    CTCiChip1Builder& setZWallWidth(float zmargin) 
    {
        m_zmargin = zmargin;
        return *this;
    }

    CTCiChip1Builder& setFileNameFor2D(const std::string& outFileName2D)
    {
        m_outFileName2D = outFileName2D;
        return *this;
    }

    CTCiChip1Builder& setFileNameFor3D(const std::string& outFileName3D)
    {
        m_outFileName3D = outFileName3D;
        return *this;
    }
    
    void build() const;
        
private:
    void generateEggSDF(const int NX, const int NY, const float xextent, const float yextent, vector<float>& sdf) const;

    void shiftRows(int rowNX, int rowNY, float rowSizeX, float rowSizeY, const SDF& rowObstacles,
                   float& padding, int& shiftedRowNX, float& shiftedRowSizeX,std::vector<SDF>& shiftedRows) const;
};

void CTCiChip1Builder::build() const
{
    if (m_ncolumns *  m_nrows *  m_nrepeat * m_resolution * m_zmargin == 0.0f || m_outFileName3D.length() == 0)
        throw std::runtime_error("Invalid parameters");
    // 1 Create 1 obstacle
    const int eggNX = static_cast<int>(m_eggSizeX * m_resolution);
    const int eggNY = static_cast<int>(m_eggSizeY * m_resolution);
    const int eggNZ = static_cast<int>(m_eggSizeZ * m_resolution);

    SDF eggSdf;
    generateEggSDF(eggNX, eggNY, m_eggSizeX, m_eggSizeY, eggSdf);

    // 2 Create 1 row of obstacles
    int rowNX = m_ncolumns*eggNX;
    int rowNY = eggNY;
    int rowSizeX = m_ncolumns * m_eggSizeX;
    int rowSizeY = m_eggSizeY;
    SDF rowObstacles;
    populateSDF(eggNX, eggNY, m_eggSizeX, m_eggSizeY, eggSdf, m_ncolumns, 1, rowObstacles);

    // 3 Shift rows
    float padding = 0.0f;
    int shiftedRowNX = 0; // they are all the same length
    float shiftedRowSizeX = 0.0f;

    std::vector<SDF> shiftedRows;
    shiftRows(rowNX, rowNY, rowSizeX, rowSizeY, rowObstacles, padding, shiftedRowNX, shiftedRowSizeX, shiftedRows);

    // 4 Collage rows
    SDF finalSDF;
    collageSDF(shiftedRowNX, rowNY, shiftedRowSizeX, rowSizeY, shiftedRows, m_nrows, true, finalSDF);

    // 5 Apply redistancing for the result
    float finalExtent[] = {shiftedRowSizeX, static_cast<float>(m_nrows * rowSizeY)};
    int finalN[] = {shiftedRowNX, m_nrows*rowNY};
    const float dx = finalExtent[0] / (finalN[0] - 1);
    const float dy = finalExtent[1] / (finalN[1] - 1);
    Redistance redistancer(0.25f * min(dx, dy), dx, dy, finalN[0], finalN[1]);
    redistancer.run(1e2, &finalSDF[0]);

    // 6 Repeat this pattern
    SDF finalSDF2;
    populateSDF(finalN[0], finalN[1], finalExtent[0], finalExtent[1], finalSDF, 1, m_nrepeat, finalSDF2);
    std::swap(finalSDF, finalSDF2);

    // 6 Write result to the file
    if (m_outFileName2D.length() != 0)
        writeDAT(m_outFileName2D, finalSDF, finalN[0], m_nrepeat * finalN[1], 1, finalExtent[0], m_nrepeat * finalExtent[1], 1.0f);

    conver2Dto3D(finalN[0], m_nrepeat * finalN[1], finalExtent[0], m_nrepeat*finalExtent[1], finalSDF,
                 eggNZ, m_eggSizeZ - 2.0f*m_zmargin, m_zmargin, m_outFileName3D);
}

void CTCiChip1Builder::generateEggSDF(const int NX, const int NY, const float xextent, const float yextent, vector<float>& sdf) const
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

void CTCiChip1Builder::shiftRows(int rowNX, int rowNY, float rowSizeX, float rowSizeY, const SDF& rowObstacles,
                                 float& padding, int& shiftedRowNX, float& shiftedRowSizeX,std::vector<SDF>& shiftedRows) const
{
    const int nRowsPerShift = static_cast<int>(ceil(m_eggSizeX / (m_eggSizeY * tan(m_angle))));
    if (fabs(m_eggSizeX / (m_eggSizeY * tan(m_angle)) - nRowsPerShift) > 1e-1) {
        throw std::runtime_error("Suggest changing the angle");
    }

    padding = float(ceil(m_nrows * m_eggSizeY * tan(m_angle)));
    // TODO Do I need this nUniqueRows?
    int nUniqueRows = m_nrows;
    if (m_nrows > nRowsPerShift) {
        nUniqueRows = nRowsPerShift;
        padding = float(round(nRowsPerShift * m_eggSizeY * tan(m_angle)));
    }

    // TODO fix this stupid workaround
    if (padding < 32.0f)
        padding = 0.0f;
    if (padding == 57.0f)
        padding = 56.0f;
    padding = padding + 8; // adjust padding to have desired size

    std::cout << "Launching rows generation. Padding = "<< padding <<std::endl;
    shiftedRows.resize(nUniqueRows);
    for (int i = 0; i < nUniqueRows; ++i) {
        float xshift = (nUniqueRows - i -1 ) * 32.0f * tan(m_angle);
        shiftSDF(rowNX, rowNY, rowSizeX, rowSizeY, rowObstacles, xshift, padding, shiftedRowNX, shiftedRowSizeX, shiftedRows[i]);
    }
}


int main(int argc, char ** argv)
{
    
    int nColumns = 5;
    int nRows = 57;
    int nRepeat = 2;

    int zMargin = 5.0f;

    std::string outFileName = "3d";

    CTCiChip1Builder builder;
    try {
        builder.setNColumns(nColumns)
               .setNRows(nRows)
               .setRepeat(nRepeat)
               .setResolution(1.0f)
               .setZWallWidth(zMargin)
               .setFileNameFor3D(outFileName)
               .build();
    } catch(const std::exception& ex) {
        std::cout << "ERROR: " << ex.what() << std::endl;
    }
    return 0;
}

