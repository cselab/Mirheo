/*
 *  collage.cpp
 *  Part of CTC/device-gen/common/
 *
 *  Created and authored by Diego Rossinelli and Kirill Lykov on 2015-03-20.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */
#pragma once

#include <vector>

void collageSDF(const int NX, const int NY, const float xextent, const float yextent, 
                const std::vector< std::vector<float> >& sampleSDF, const int ytimes,
                bool wallInY, std::vector<float>& outputSDF);

void populateSDF(const int NX, const int NY, const float xextent, const float yextent, const std::vector<float>& sampleSDF,
                 const int xtimes, const int ytimes, std::vector<float>& outputSDF);

void shiftSDF(const int NX, const int NY, const float xextent, const float yextent, const std::vector<float>& inputGrid, 
              const float xshift, const float xpadding, int& newNX, float& newXextent, std::vector<float>& outGrid);
