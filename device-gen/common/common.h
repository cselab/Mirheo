/*
 *  common.h
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
#include <string>
#include <vector>
#include <cstdio>
#include <assert.h>
#include <cmath>

inline float sign(float x) {
    return 1.0f - 2.0f*std::signbit(x);
}

inline void readDAT(const std::string& fileName, std::vector<float>& data, 
        int& NX, int& NY, int& NZ, float& xextent, float& yextent, float& zextent)
{
    FILE * f = fopen(fileName.c_str(), "r");
    assert(f != 0);
    int result = fscanf(f, "%f %f %f\n", &xextent, &yextent, &zextent);
    assert(result == 3);
    result = fscanf(f, "%d %d %d\n", &NX, &NY, &NZ);
    assert(result == 3);
    printf("Read file %s. Extent: [%f, %f, %f]. Grid size: [%d, %d, %d]\n", 
            fileName.c_str(), xextent, yextent, zextent, NX, NY, NZ);
    data.resize(NX * NY * NZ, 0.0f);
    result = fread(&data[0], sizeof(float), NX * NY * NZ, f);
    if (result != data.size()) {    
        printf("ERROR: read less than expected");
        exit(3);
    }
    fclose(f);
}

inline void writeDAT(const std::string& fileName,  std::vector<float>& data,
        const int NX, const int NY, const int NZ, float xextent, float yextent, float zextent)
{
    FILE * f = fopen(fileName.c_str(), "w");
    assert(f != 0);
    fprintf(f, "%f %f %f\n", xextent, yextent, zextent);
    fprintf(f, "%d %d %d\n", NX, NY, NZ);
    
    unsigned char * ptr = (unsigned char *)&data[0];
    if ((ptr[0] >= 9 && ptr[0] <= 13) || ptr[0] == 32 )
    {
        ptr[0] = (ptr[0] == 32) ? 33 : (ptr[0] < 11) ? 8 : 14;
        printf("INFO: some symbols were changed while writing\n");
    }
    
    int result = fwrite(&data[0], sizeof(float), (int)data.size(), f);
    if (result != data.size()) {
        printf("ERROR: written less than expected");
        exit(3);
    }
        
    fclose(f);
}

