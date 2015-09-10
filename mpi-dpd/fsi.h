/*
 *  rbc-interactions.h
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

class ComputeFSI : public SoluteExchange::Visitor
{
    //TODO: use cudaEvent_t evuploaded;

    SolventWrap wsolvent;

    Logistic::KISS local_trunk;

public:

    void bind_solvent(SolventWrap wrap) { wsolvent = wrap; }

    ComputeFSI(MPI_Comm comm);

    void bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream);

    /*override of SoluteExchange::Visitor::halo*/
    void halo(ParticlesWrap solutes[26], cudaStream_t stream);
};
