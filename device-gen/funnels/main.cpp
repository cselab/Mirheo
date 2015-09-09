/*
 *  main.cpp
 *  Part of CTC/device-gen/sdf-unit-par/
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
#include <iostream>
#include <vector>
#include <argument-parser.h>
#include "../common/device-builder.h"
#include "../common/common.h"
#include "../common/collage.h"
#include "../common/redistance.h"
#include "../common/2Dto3D.h"

using namespace std;

struct Parabola 
{
    float x0;
    float ymax; // length of obstacle, for cutting the pick
    float y0;

    Parabola(float gap, float xextent) : ymax(48.0) 
    {
        x0 = xextent/2.0f - gap/2.0f;
        if (gap > 9.0)
            y0 = ymax;
        else
            y0 = ymax + 6.5;
    }    

    void line1(vector<float>& vx, vector<float>& vy) {
        int N = 500;
        float dx = 2.0f * fabs(x0) / (N - 1);
        for (int i = 0; i < N; ++i) {
            float x = i * dx - x0;
            float y = 0.0f;
            vx.push_back(x);
            vy.push_back(y);
        }
    }

    void line2(vector<float>& vx, vector<float>& vy) {
        int N = 1500;
        float dx = 2.0f * fabs(x0) / (N - 1);
        float alpha = -y0 / (x0 * x0);
        for (int i = 0; i < N; ++i) {    
            float x = i * dx - x0;
            float y = min(ymax, alpha * x*x + y0);
            vx.push_back(x);
            vy.push_back(y);
        }
    }
};

class FunnelsBuilder : public DeviceBuilder
{
    float m_gapSpace; // unit gap between obstacles
public:
    FunnelsBuilder()
    : DeviceBuilder(24.0f, 96.0f, 58.0f), m_gapSpace(1.0f)
    {}
    
    FunnelsBuilder& setNColumns(int ncolumns)
    {
        m_ncolumns = ncolumns;
        return *this;
    }

    FunnelsBuilder& setNRows(int nrows)
    {
        m_nrows = nrows;
        return *this;
    }

    FunnelsBuilder& setResolution(float resolution)
    {
        m_resolution = resolution;
        return *this;
    }

    FunnelsBuilder& setZWallWidth(float zmargin)
    {
        m_zmargin = zmargin;
        return *this;
    }

    FunnelsBuilder& setFileNameFor2D(const std::string& outFileName2D)
    {
        m_outFileName2D = outFileName2D;
        return *this;
    }

    FunnelsBuilder& setFileNameFor3D(const std::string& outFileName3D)
    {
        m_outFileName3D = outFileName3D;
        return *this;
    }

    void build();

private:
    void generateUnitSDF(float gap, vector<float>& sdf) const;
};

void FunnelsBuilder::generateUnitSDF(float gap, vector<float>& sdf) const
{
    assert(m_unitNX * m_unitNY * m_unitNZ != 0);
    vector<float> xs, ys;
    Parabola par(gap, m_unitSizeX);
    par.line1(xs, ys);
    par.line2(xs, ys);   

    const float xlb = -m_unitSizeX/2.0f;
    const float ylb = -(m_unitSizeY - par.ymax)/2.0f; 
    
    sdf.resize(m_unitNX * m_unitNY, 0.0f);
    const float dx = m_unitSizeX / (m_unitNX - 1); //TODO NX-1
    const float dy = m_unitSizeY / (m_unitNY - 1);
    const int nsamples = xs.size();
    
    for(int iy = 0; iy < m_unitNY; ++iy)
    for(int ix = 0; ix < m_unitNX; ++ix)
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
            const float alpha = -par.y0 / (par.x0 * par.x0);
            const float ycurve = min(par.ymax, alpha * x*x + par.y0);
            
            if (x >= -par.x0 && x <= par.x0 && y >= 0 && y <= ycurve)
                s = +1;
        }

        
        sdf[ix + m_unitNX * iy] = s * sqrt(distance2);
    }
}

void FunnelsBuilder::build()
{
    if (m_ncolumns *  m_nrows * m_resolution * m_zmargin == 0.0f || m_outFileName3D.length() == 0)
        throw std::runtime_error("Invalid parameters");

    // 1 Create obstacles with different gaps
    m_unitNX = static_cast<int>(m_unitSizeX * m_resolution);
    m_unitNY = static_cast<int>(m_unitSizeY * m_resolution);
    m_unitNZ = static_cast<int>(m_unitSizeZ * m_resolution); 
 
    std::vector<SDF> unitSDF(m_nrows);
    for (int i = 0; i < m_nrows; ++i) {
        float gap = m_gapSpace * (i + 3);
        generateUnitSDF(gap, unitSDF[i]);
    }

    // 2 Create rows of obstacles
    std::vector<SDF> rows(m_nrows);
    for (int i = 0; i < m_nrows; ++i) {
        populateSDF(m_unitNX, m_unitNY, m_unitSizeX, m_unitSizeY, unitSDF[i], m_ncolumns, 1, rows[i]);
    }    

    // 3 Collage rows
    SDF finalSDF;
    collageSDF(m_unitNX * m_ncolumns, m_unitNY, m_unitSizeX * m_ncolumns, m_unitSizeY, rows, m_nrows, false, finalSDF);

    // 4 Apply redistancing for the result
    float finalExtent[] = {m_unitSizeX * m_ncolumns, m_unitSizeY * m_nrows};
    int finalN[] = {m_unitNX * m_ncolumns, m_unitNY * m_nrows};
    const float dx = finalExtent[0] / (finalN[0] - 1);
    const float dy = finalExtent[1] / (finalN[1] - 1);
    Redistance redistancer(0.25f * min(dx, dy), dx, dy, finalN[0], finalN[1]);
    redistancer.run(m_niterRedistance, &finalSDF[0]);
    
    if (m_outFileName2D.length() != 0)
        writeDAT(m_outFileName2D.c_str(), finalSDF, finalExtent[0], finalExtent[1], 1.0f, finalN[0], finalN[1], 1);
    
    conver2Dto3D(finalN[0], finalN[1], finalExtent[0], finalExtent[1], finalSDF, m_unitNZ, m_unitSizeZ - 2.0f*m_zmargin, m_zmargin, m_outFileName3D);
}

int main(int argc, char ** argv)
{
    ArgumentParser argp(vector<string>(argv, argv + argc));

    int nColumns = argp("-nColumns").asInt(1);
    int nRows = argp("-nRows").asInt(1);
    float zMargin = static_cast<float>(argp("-zMargin").asDouble(5.0));
    float resolution = static_cast<float>(argp("-zResolution").asDouble(1.0));

    std::string outFileName = argp("-out").asString("3d");

    FunnelsBuilder builder;
    try {
        builder.setNColumns(nColumns)
               .setNRows(nRows)
               .setResolution(1.0f)
               .setZWallWidth(zMargin)
               .setFileNameFor3D(outFileName)
               .setResolution(resolution)
               .build();
    } catch(const std::exception& ex) {
        std::cout << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
