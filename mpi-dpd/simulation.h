#pragma once

#include <csignal>
#include <map>
#include <vector>
#include <string>

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
    ParticleArray particles;
    
    CellLists cells;
    CollectionRBC * rbcscoll;
    CollectionCTC * ctcscoll;

    H5PartDump dump_part, *dump_part_solvent;
    H5FieldDump dump_field;
    
    RedistributeParticles redistribute;
    RedistributeRBCs redistribute_rbcs;
    RedistributeCTCs redistribute_ctcs;
    
    ComputeInteractionsDPD dpd;
    ComputeInteractionsRBC rbc_interactions;
    ComputeInteractionsCTC ctc_interactions;
    ComputeInteractionsWall * wall;

    bool (*check_termination)();

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
    void _data_dump(const int idtimestep);
    void _update_and_bounce();

public:

    Simulation(MPI_Comm cartcomm, MPI_Comm activecomm, bool (*check_termination)()) ;
    
    void run();

    ~Simulation();
};
