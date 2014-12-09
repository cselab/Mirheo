#pragma once

#include <vector>

#include "common.h"

struct ParticleArray
{
    int size;

    SimpleDeviceBuffer<Particle> xyzuvw;
    SimpleDeviceBuffer<Acceleration> axayaz;

    ParticleArray() {}
    
    ParticleArray(std::vector<Particle> ic);

    void resize(int n);
    void update_stage1(const float gradpressure[3]);
    void update_stage2_and_1(const float gradpressure[3]);
};

class CollectionRBC : ParticleArray
{
    MPI_Comm cartcomm;
    
    int nrbcs, L, myrank, dims[3], periods[3], coords[3];
    
public:
    
    int nvertices;

    CollectionRBC(MPI_Comm cartcomm, const int L);
    
    void update_stage1();
    void update_stage2_and_1();
     
    Particle * data() { return xyzuvw.data; }
    Acceleration * acc() { return axayaz.data; }
    void remove(const int * const entries, const int nentries);
    void resize(const int rbcs_count);
    
    int count() { return nrbcs; }
    int pcount() { return nrbcs * nvertices; }
    
    void dump(MPI_Comm comm);
};

