/*
 *  simulation.h
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once

#include <csignal>
#include <map>
#include <vector>
#include <string>

#include <pthread.h>

#ifdef _USE_NVTX_
#include <cuda_profiler_api.h>
#endif

#include "common.h"
#include "containers.h"
#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "rbc-interactions.h"
#include "ctc.h"
#include "io.h"

class Simulation
{
    SimpleDeviceBuffer<Particle> unordered_particles;
    ParticleArray particles;
    
    CellLists cells;
    CollectionRBC * rbcscoll;
    CollectionCTC * ctcscoll;
    
    RedistributeParticles redistribute;
    RedistributeRBCs redistribute_rbcs;
    RedistributeCTCs redistribute_ctcs;
    
    ComputeInteractionsDPD dpd;
    ComputeInteractionsRBC rbc_interactions;
    ComputeInteractionsCTC ctc_interactions;
    ComputeInteractionsWall * wall;

    LocalComm localcomm;

    bool (*check_termination)();
    bool simulation_is_done;

    MPI_Comm activecomm, cartcomm;

    cudaStream_t mainstream;
    
    std::map<std::string, double> timings;

    const size_t nsteps;
    float driving_acceleration;
    float host_idle_time;
    int nranks, rank;  
	    
    std::vector<Particle> _ic();

    void _redistribute();
    void _report(const bool verbose, const int idtimestep);
    void _create_walls(const bool verbose, bool & termination_request);
    void _remove_bodies_from_wall(CollectionRBC * coll);
    void _forces();
    void _datadump(const int idtimestep);
    void _update_and_bounce();
    void _lockstep();

    pthread_t thread_datadump;
    pthread_mutex_t mutex_datadump;
    pthread_cond_t request_datadump, done_datadump;
    bool datadump_pending;
    int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs, datadump_nctcs;

    PinnedHostBuffer<Particle> particles_datadump;
    PinnedHostBuffer<Acceleration> accelerations_datadump;

    cudaEvent_t evdownloaded;

    void  _datadump_async();

public:

    Simulation(MPI_Comm cartcomm, MPI_Comm activecomm, bool (*check_termination)()) ;
    
    void run();

    ~Simulation();

    static void * datadump_trampoline(void * x) { ((Simulation *)x)->_datadump_async(); return NULL; }
};
