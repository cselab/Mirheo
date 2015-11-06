/*
 *  contact.h
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <vector>
#include <rbc-cuda.h>

#include "solute-exchange.h"

#include <../dpd-rng.h>

class ComputeContact : public SoluteExchange::Visitor
{
    std::vector<ParticlesWrap> wsolutes;
    
    SimpleDeviceBuffer<uchar4> subindices;
    SimpleDeviceBuffer<unsigned char> compressed_cellscount;
    SimpleDeviceBuffer<int> cellsentries, cellsstart, cellscount;
    SimpleDeviceBuffer<Particle> allhalos;
    SimpleDeviceBuffer<Acceleration> allhalosacc;

    Logistic::KISS local_trunk;
    
public:

    ComputeContact(MPI_Comm comm);

    void attach_bulk(std::vector<ParticlesWrap> wsolutes) { this->wsolutes = wsolutes; }

    /*override of SoluteExchange::Visitor::halo*/
    void halo(ParticlesWrap solutes[26], cudaStream_t stream);
};
