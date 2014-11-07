#pragma once

#include <mpi.h>


#include "common.h"


//particles may fall outside my subdomain. i might loose old particles and receive new ones.
//redistribution is performed in 2 stages and unfortunately the first stage is stateful (tmp vector here below)
//that's why i need a class.
class RedistributeParticles
{
    MPI_Comm cartcomm;
    
    int L, myrank, dims[3], periods[3], coords[3], rankneighbors[27], domain_extent[3];
    int leaving_start[28], arriving_start[28], notleaving, arriving;

    std::vector<Particle> tmp;

    MPI_Request sendreq[27], recvreq[27];
    
public:
    
    RedistributeParticles(MPI_Comm cartcomm, int L);
    
    int stage1(Particle * p, int n);

    int stage2(Particle * p, int n);
};
