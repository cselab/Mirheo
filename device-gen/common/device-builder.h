
/*
 *  device-builder.h
 *  Part of CTC/device-gen/ctc-ichip/
 *
 *  Created and authored by Kirill Lykov on 2015-09-7.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

class DeviceBuilder
{
protected:
    typedef vector<float> SDF;
    int m_ncolumns, m_nrows;
    float m_resolution, m_zmargin;
    float m_unitSizeX, m_unitSizeY, m_unitSizeZ; // size of the egg with the empty space aroung it
    std::string m_outFileName2D, m_outFileName3D;
    int m_niterRedistance;
    int m_unitNX, m_unitNY, m_unitNZ;
public:
    DeviceBuilder(float unitSizeX, float unitSizeY, float unitSizeZ)
    : m_ncolumns(0), m_nrows(0), m_resolution(0), m_zmargin(0),
      m_unitSizeX(unitSizeX), m_unitSizeY(unitSizeY), m_unitSizeZ(unitSizeZ),
      m_niterRedistance(1e3), m_unitNX(0), m_unitNY(0), m_unitNZ(0)
    {}

    virtual void build() = 0;

    virtual ~DeviceBuilder() {}
};
