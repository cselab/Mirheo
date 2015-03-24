/*
 *  main.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-14.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>
#include <csignal>
#include <mpi.h>
#include <errno.h>

#include "simulation.h"

bool currently_profiling = false;

namespace SignalHandling
{
    volatile sig_atomic_t graceful_exit = 0, graceful_signum = 0;
    sigset_t mask;
    
    void signal_handler(int signum)
    {
	graceful_exit = 1;
	graceful_signum = signum;
    }
    
    void setup()
    {
	sigemptyset (&mask);
	sigaddset (&mask, SIGINT);
	
	if (sigprocmask(SIG_BLOCK, &mask, NULL) < 0) 
	{
	    perror ("sigprocmask");
	    exit(EXIT_FAILURE);
	}
	
	struct sigaction action;
	memset(&action, 0, sizeof(struct sigaction));
	action.sa_handler = signal_handler;
	sigaction(SIGUSR1, &action, NULL);
    }
    
    bool check_termination_request()
    {
	if (graceful_exit)
	    return true;
	
	struct timespec timeout;
	timeout.tv_sec = 0;
	timeout.tv_nsec = 1000;
	
	return sigtimedwait(&mask, NULL, &timeout) >= 0;
    }   
}

int main(int argc, char ** argv)
{
    int ranks[3];

    //parsing of the positional arguments
    if (argc != 4)
    {
	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
	exit(-1);
    }
    else
    	for(int i = 0; i < 3; ++i)
	    ranks[i] = atoi(argv[1 + i]);

    SignalHandling::setup();
    
    CUDA_CHECK(cudaSetDevice(0));

    CUDA_CHECK(cudaDeviceReset());
    
    int nranks, rank;   
    
    {
	MPI_CHECK( MPI_Init(&argc, &argv) );

	MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	
	MPI_Comm activecomm = MPI_COMM_WORLD;

	bool reordering = true;

	const char * env_reorder = getenv("MPICH_RANK_REORDER_METHOD");

	//reordering of the ranks according to the computational domain and environment variables
	if (atoi(env_reorder ? env_reorder : "-1") == atoi("3"))
	{
	    reordering = false;

	    const bool usefulrank = rank < ranks[0] * ranks[1] * ranks[2];
	    
	    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, usefulrank, rank, &activecomm)) ;

	    MPI_CHECK(MPI_Barrier(activecomm));

	    if (!usefulrank)
	    {
		printf("rank %d has been thrown away\n", rank);
		fflush(stdout);

		MPI_CHECK(MPI_Barrier(activecomm));

		MPI_Finalize();

		return 0;
	    }

	    MPI_CHECK(MPI_Barrier(activecomm));
	}
	
	MPI_Comm cartcomm;

	int periods[] = {1, 1, 1};	    

	MPI_CHECK( MPI_Cart_create(activecomm, 3, ranks, periods, (int)reordering, &cartcomm) );

	activecomm = cartcomm;

	//print the rank-to-node mapping
	{
	    char name[1024];
	    int len;
	    MPI_CHECK(MPI_Get_processor_name(name, &len));
	    
	    int dims[3], periods[3], coords[3];
	    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

	    MPI_CHECK(MPI_Barrier(activecomm));

	    printf("RANK %d: (%d, %d, %d) -> %s\n", rank, coords[0], coords[1], coords[2], name);
	    fflush(stdout);

	    MPI_CHECK(MPI_Barrier(activecomm));
	}
	
	//RAII
	{
	    Simulation simulation(cartcomm, activecomm, SignalHandling::check_termination_request);

	    simulation.run();
	}
	
	if (activecomm != cartcomm)
	    MPI_CHECK(MPI_Comm_free(&activecomm));
	
	MPI_CHECK(MPI_Comm_free(&cartcomm));
	
	MPI_CHECK(MPI_Finalize());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
