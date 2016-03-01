/*
 *  simulation.h
 *  Part of uDeviceX/mpi-dpd/
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
#include "dpd.h"
#include "wall.h"
#include "fsi.h"
#include "contact.h"
#include "solute-exchange.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "ctc.h"
#include "io.h"
#include "velcontroller.h"
#include "velsampler.h"

class Simulation
{
    ParticleArray particles_pingpong[2];
    ParticleArray * particles, * newparticles;
    SimpleDeviceBuffer<float4> xyzouvwo;
    SimpleDeviceBuffer<ushort4> xyzo_half;
    SimpleDeviceBuffer<float> stresses[6];
    
    CellLists cells;
    CollectionRBC * rbcscoll;
    CollectionCTC * ctcscoll;
    
    RedistributeParticles redistribute;
    RedistributeRBCs redistribute_rbcs;
    RedistributeCTCs redistribute_ctcs;
    
    ComputeDPD dpd;
    SoluteExchange solutex;
    ComputeFSI fsi;
    ComputeContact contact;

    ComputeWall * wall;

    bool (*check_termination)();
    bool simulation_is_done;

    MPI_Comm activecomm, cartcomm, intercomm;
    //LocalComm localcomm;

    cudaStream_t mainstream, uploadstream, downloadstream;
    
    std::map<std::string, double> timings;

    const size_t nsteps;
    float driving_acceleration;
    float host_idle_time;
    int nranks, rank;
	    
    std::vector<Particle> _ic();
    void _update_helper_arrays();
    
    void _redistribute();
    void _report(const bool verbose, const int idtimestep);
    void _create_walls(const bool verbose, bool & termination_request);
    void _remove_bodies_from_wall(CollectionRBC * coll);
    void _forces(bool firsttime = false);
    void _datadump(const int idtimestep);
    void _update_and_bounce();
    void _lockstep();

    pthread_t thread_datadump;
    pthread_mutex_t mutex_datadump;
    pthread_cond_t request_datadump, done_datadump;
    bool datadump_pending;
    int datadump_idtimestep, datadump_nsolvent, datadump_nrbcs, datadump_nctcs;
    bool async_thread_initialized;

    PinnedHostBuffer<Particle> particles_datadump;
    PinnedHostBuffer<Acceleration> accelerations_datadump;
    PinnedHostBuffer<float> stresses_datadump[6];

    cudaEvent_t evdownloaded;

    void  _datadump_async();

    VelController* velcontrol;
    VelSampler*    velsampler;

public:

    Simulation(MPI_Comm cartcomm, MPI_Comm activecomm, MPI_Comm intercomm, bool (*check_termination)()) ;
    
    void run();

    ~Simulation();
};
