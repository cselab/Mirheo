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

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <vector>
#include <map>

#include <cuda_profiler_api.h>

#include "common.h"
#include "io.h"
#include "containers.h"
#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "rbc-interactions.h"
#include "ctc.h"

bool currently_profiling = false;
static const int blocksignal = false;

volatile sig_atomic_t graceful_exit = 0, graceful_signum = 0;
sigset_t mask;

void signal_handler(int signum)
{
    graceful_exit = 1;
    graceful_signum = signum;
}

using namespace std;

int main(int argc, char ** argv)
{
    int ranks[3];

    if (argc != 4)
    {
	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
	exit(-1);
    }
    else
    	for(int i = 0; i < 3; ++i)
	    ranks[i] = atoi(argv[1 + i]);

    if (blocksignal)
    {
	sigemptyset (&mask);
	sigaddset (&mask, SIGUSR1);
	sigaddset (&mask, SIGINT);
	
	if (sigprocmask(SIG_BLOCK, &mask, NULL) < 0) 
	{
	    perror ("sigprocmask");
	    return 1;
	}
    }
    else
    {
	struct sigaction action;
	memset(&action, 0, sizeof(struct sigaction));
	action.sa_handler = signal_handler;
	//sigaction(SIGINT, &action, NULL);
	sigaction(SIGUSR1, &action, NULL);
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset()); //WILL THIS MESS UP WITH MPS? 
    
    int nranks, rank;   
    
    {
	MPI_CHECK( MPI_Init(&argc, &argv) );

	MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	
	MPI_Comm activecomm = MPI_COMM_WORLD;
	bool reordering = true;
	const char * env_reorder = getenv("MPICH_RANK_REORDER_METHOD");

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

	srand48(rank);
	
	MPI_Comm cartcomm;
	int periods[] = {1, 1, 1};	    
	MPI_CHECK( MPI_Cart_create(activecomm, 3, ranks, periods, (int)reordering, &cartcomm) );
	activecomm = cartcomm;

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
	
	{
	    vector<Particle> ic(XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN * numberdensity);
	    
	    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
	    for(int i = 0; i < ic.size(); ++i)
		for(int c = 0; c < 3; ++c)
		{
		    ic[i].x[c] = -L[c] * 0.5 + drand48() * L[c];
		    ic[i].u[c] = 0;
		}
	    	    	 	    
	    ParticleArray particles(ic);
	    
	    CellLists cells(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN);		  
	    CollectionRBC * rbcscoll = NULL;
	    
	    if (rbcs)
	    {
		rbcscoll = new CollectionRBC(cartcomm);
		rbcscoll->setup();
	    }

	    CollectionCTC * ctcscoll = NULL;

	    if (ctcs) 
	    {
		ctcscoll = new CollectionCTC(cartcomm);
		ctcscoll->setup();
	    }

	    H5PartDump dump_part("allparticles.h5part", activecomm, cartcomm), *dump_part_solvent = NULL;
	    H5FieldDump dump_field(cartcomm);

	    RedistributeParticles redistribute(cartcomm);
	    RedistributeRBCs redistribute_rbcs(cartcomm);
	    RedistributeCTCs redistribute_ctcs(cartcomm);

	    ComputeInteractionsDPD dpd(cartcomm);
	    ComputeInteractionsRBC rbc_interactions(cartcomm);
	    ComputeInteractionsCTC ctc_interactions(cartcomm);
	    ComputeInteractionsWall * wall = NULL;

            //Side not of Yu-Hang:
	    //in production runs replace the numbers with 4 unique ones that are same across ranks
            //KISS rng_trunk( 0x26F94D92, 0x383E7D1E, 0x46144B48, 0x6DDD73CB );
	    
	    cudaStream_t mainstream;
	    CUDA_CHECK(cudaStreamCreate(&mainstream));
	    	    
	    CUDA_CHECK(cudaPeekAtLastError());

	    cells.build(particles.xyzuvw.data, particles.size, mainstream);

	    dpd.pack(particles.xyzuvw.data, particles.size, cells.start, cells.count, mainstream);
	    dpd.local_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, mainstream);
	    dpd.consolidate_and_post(particles.xyzuvw.data, particles.size, mainstream);
	    dpd.wait_for_messages(mainstream);
	    dpd.remote_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, mainstream);

	    if (rbcscoll)
		rbc_interactions.evaluate(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
					  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);

	    if (ctcscoll)
		ctc_interactions.evaluate(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
					  ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);

	    float driving_acceleration = 0;

	    if (!walls && pushtheflow)
		driving_acceleration = hydrostatic_a;		    

	    std::map<string, double> timings;
	    float host_idle_time = 0;

	    const size_t nsteps = (int)(tend / dt);

	    if (rank == 0 && !walls)
		printf("the simulation begins now and it consists of %.3e steps\n", (double)nsteps);

	    double time_simulation_start = MPI_Wtime();

	    int it;

	    for(it = 0; it < nsteps; ++it)
	    {
#ifdef _USE_NVTX_
		if (it == 1001)
		{
		    currently_profiling = true;
		    CUDA_CHECK(cudaProfilerStart());
		    
		}
		else if (it == 1051)
		{
		    CUDA_CHECK(cudaProfilerStop());
		    currently_profiling = false;
		    CUDA_CHECK(cudaDeviceSynchronize());
		    break;
		}
#endif
	
		if (it % steps_per_report == 0)
		{ 
		    CUDA_CHECK(cudaStreamSynchronize(mainstream));

		    //check for termination requests
		    if (blocksignal)
		    {
			struct timespec timeout;
			timeout.tv_sec = 0;
			timeout.tv_nsec = 1000;
			
			graceful_exit = sigtimedwait(&mask, NULL, &timeout) >= 0;
		    }
		    else
		    {
			int exitrequest = graceful_exit;
			MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &exitrequest, 1, MPI_LOGICAL, MPI_LOR, activecomm)); 
			graceful_exit = exitrequest;
		    }

		    if (graceful_exit)
		    {
			if (!rank)
			    printf("Got a termination request. Time to save and exit!\n");
		    
			break;
		    }	    		    

		    report_host_memory_usage(activecomm, stdout);
		   
		    {
			static double t0 = MPI_Wtime(), t1;

			t1 = MPI_Wtime();

			float host_busy_time = (MPI_Wtime() - t0) - host_idle_time;
			
			host_busy_time *= 1e3 / steps_per_report;

			float sumval, maxval, minval;
			MPI_CHECK(MPI_Reduce(&host_busy_time, &sumval, 1, MPI_FLOAT, MPI_SUM, 0, activecomm));
			MPI_CHECK(MPI_Reduce(&host_busy_time, &maxval, 1, MPI_FLOAT, MPI_MAX, 0, activecomm));
			MPI_CHECK(MPI_Reduce(&host_busy_time, &minval, 1, MPI_FLOAT, MPI_MIN, 0, activecomm));
			
			int commsize;
			MPI_CHECK(MPI_Comm_size(activecomm, &commsize));
			
			const double imbalance = 100 * (maxval / sumval * commsize - 1);

			if (it > 0 && rank == 0 && imbalance > 5)
			    printf("\x1b[93moverall imbalance: %.f%%, host workload min/avg/max: %.2f/%.2f/%.2f ms\x1b[0m\n", 
				   imbalance , minval, sumval / commsize, maxval);
			
			host_idle_time = 0;
			t0 = t1;
		    }
 
		    {
			static double t0 = MPI_Wtime(), t1;

			t1 = MPI_Wtime();
		    
			if (it > 0 && rank == 0)
			{
			    printf("\x1b[92mbeginning of time step %d (%.3f ms)\x1b[0m\n", it, (t1 - t0) * 1e3 / steps_per_report);
			    printf("in more details, per time step:\n");
			    double tt = 0;
			    for(std::map<string, double>::iterator it = timings.begin(); it != timings.end(); ++it)
			    {
				printf("%s: %.3f ms\n", it->first.c_str(), it->second * 1e3 / steps_per_report);
				tt += it->second;
				it->second = 0;
			    }
			    printf("discrepancy: %.3f ms\n", ((t1 - t0) - tt) * 1e3 / steps_per_report);
			}

			t0 = t1;
		    }
		}
		
		double tstart;
		
		if (it == 0)
		{
		    particles.update_stage1(driving_acceleration, mainstream);
		    
		    if (rbcscoll)
			rbcscoll->update_stage1(driving_acceleration, mainstream);

		    if (ctcscoll)
			ctcscoll->update_stage1(driving_acceleration, mainstream);
		}
		
		tstart = MPI_Wtime();
		const int newnp = redistribute.stage1(particles.xyzuvw.data, particles.size, mainstream, host_idle_time);
		particles.resize(newnp);
		redistribute.stage2(particles.xyzuvw.data, particles.size, mainstream, host_idle_time);
		timings["redistribute-particles"] += MPI_Wtime() - tstart;
		
		CUDA_CHECK(cudaPeekAtLastError());

		if (rbcscoll)
		{	
		    tstart = MPI_Wtime();
		    const int nrbcs = redistribute_rbcs.stage1(rbcscoll->data(), rbcscoll->count(), mainstream);
		    rbcscoll->resize(nrbcs);
		    redistribute_rbcs.stage2(rbcscoll->data(), rbcscoll->count(), mainstream);
		    timings["redistribute-rbc"] += MPI_Wtime() - tstart;
		}
		
		CUDA_CHECK(cudaPeekAtLastError());
			
		if (ctcscoll)
		{	
		    tstart = MPI_Wtime();
		    const int nctcs = redistribute_ctcs.stage1(ctcscoll->data(), ctcscoll->count(), mainstream);
		    ctcscoll->resize(nctcs);
		    redistribute_ctcs.stage2(ctcscoll->data(), ctcscoll->count(), mainstream);
		    timings["redistribute-ctc"] += MPI_Wtime() - tstart;
		}
		
		CUDA_CHECK(cudaPeekAtLastError());
				
		//create the wall when it is time
		if (walls && it >= wall_creation_stepid && wall == NULL)
		{
		    if (rank == 0)
			printf("creation of the walls...\n");

		    CUDA_CHECK(cudaDeviceSynchronize());

		    int nsurvived = 0;
		    ExpectedMessageSizes new_sizes;
		    wall = new ComputeInteractionsWall(cartcomm, particles.xyzuvw.data, particles.size, nsurvived, new_sizes);
		    
		    //adjust the message sizes if we're pushing the flow in x
		    {
			const double xvelavg = getenv("XVELAVG") ? atof(getenv("XVELAVG")) : pushtheflow;
			const double yvelavg = getenv("YVELAVG") ? atof(getenv("YVELAVG")) : 0;
			const double zvelavg = getenv("ZVELAVG") ? atof(getenv("ZVELAVG")) : 0;
		
			for(int code = 0; code < 27; ++code)
			{
			    const int d[3] = {
				(code % 3) - 1,
				((code / 3) % 3) - 1,
				((code / 9) % 3) - 1
			    };
			    
			    const double IudotnI = 
				fabs(d[0] * xvelavg) + 
				fabs(d[1] * yvelavg) + 
				fabs(d[2] * zvelavg) ;

			    const float factor = 1 + IudotnI * dt * 10 * numberdensity;
			    
			    //printf("RANK %d: direction %d %d %d -> IudotnI is %f and final factor is %f\n",
			    //rank, d[0], d[1], d[2], IudotnI, 1 + IudotnI * dt * numberdensity);
			    
			    new_sizes.msgsizes[code] *= factor;
			}
		    }


		    MPI_CHECK(MPI_Barrier(activecomm));
		    redistribute.adjust_message_sizes(new_sizes);
		    dpd.adjust_message_sizes(new_sizes);
		    MPI_CHECK(MPI_Barrier(activecomm));

		    if (hdf5part_dumps)
			dump_part.close();
			
		    //there is no support for killing zero-workload ranks for rbcs and ctcs just yet
		    if (!rbcs && !ctcs)
		    {
			const bool local_work = new_sizes.msgsizes[1 + 3 + 9] > 0;
			
			MPI_CHECK(MPI_Comm_split(cartcomm, local_work, rank, &activecomm)) ;
			
			MPI_CHECK(MPI_Comm_rank(activecomm, &rank));
			
			if (!local_work )
			{
			    if (rank == 0)
			    {
				int nkilled;
				MPI_CHECK(MPI_Comm_size(activecomm, &nkilled));
				
				printf("THERE ARE %d RANKS WITH ZERO WORKLOAD THAT WILL MPI-FINALIZE NOW.\n", nkilled);
			    } 
			    
			    break;
			}
		    }
		    
		    if (hdf5part_dumps)
			dump_part_solvent = new H5PartDump("solvent-particles.h5part", activecomm, cartcomm);

		    particles.resize(nsurvived);
		    particles.clear_velocity();
	
		    {
			H5PartDump sd("survived-particles.h5part", activecomm, cartcomm);
			Particle * p = new Particle[particles.size];
			
			CUDA_CHECK(cudaMemcpy(p, particles.xyzuvw.data, sizeof(Particle) * particles.size, cudaMemcpyDeviceToHost));
			
			sd.dump(p, particles.size);
			
			delete [] p;
		    }
	    		    
		    if (rank == 0)
		    {
			if( access( "particles.xyz", F_OK ) != -1 )
			{
			    const int retval = rename ("particles.xyz", "particles-equilibration.xyz");
			    assert(retval != -1);
			}
		    
			if( access( "rbcs.xyz", F_OK ) != -1 )  
			{
			    const int retval = rename ("rbcs.xyz", "rbcs-equilibration.xyz");
			    assert(retval != -1);
			}
		    }
		    
		    CUDA_CHECK(cudaPeekAtLastError());

		    //remove rbcs touching the wall
		    if(rbcscoll && rbcscoll->count())
		    {
			SimpleDeviceBuffer<int> marks(rbcscoll->pcount());
			
			SolidWallsKernel::fill_keys<<< (rbcscoll->pcount() + 127) / 128, 128 >>>(rbcscoll->data(), rbcscoll->pcount(), marks.data);
			
			vector<int> tmp(marks.size);
			CUDA_CHECK(cudaMemcpy(tmp.data(), marks.data, sizeof(int) * marks.size, cudaMemcpyDeviceToHost));
			
			const int nrbcs = rbcscoll->count();
			const int nvertices = rbcscoll->nvertices;

			std::vector<int> tokill;
			for(int i = 0; i < nrbcs; ++i)
			{
			    bool valid = true;

			    for(int j = 0; j < nvertices && valid; ++j)
				valid &= 0 == tmp[j + nvertices * i];
			    
			    if (!valid)
				tokill.push_back(i);
			}

			rbcscoll->remove(&tokill.front(), tokill.size());
			rbcscoll->clear_velocity();
		    }

		    CUDA_CHECK(cudaPeekAtLastError());

		    //remove ctcs touching the wall
		    if(ctcscoll && ctcscoll->count())
		    {
			SimpleDeviceBuffer<int> marks(ctcscoll->pcount());
			
			SolidWallsKernel::fill_keys<<< (ctcscoll->pcount() + 127) / 128, 128 >>>(ctcscoll->data(), ctcscoll->pcount(), marks.data);
			
			vector<int> tmp(marks.size);
			CUDA_CHECK(cudaMemcpy(tmp.data(), marks.data, sizeof(int) * marks.size, cudaMemcpyDeviceToHost));
			
			const int nctcs = ctcscoll->count();
			const int nvertices = ctcscoll->nvertices;

			std::vector<int> tokill;
			for(int i = 0; i < nctcs; ++i)
			{
			    bool valid = true;

			    for(int j = 0; j < nvertices && valid; ++j)
				valid &= 0 == tmp[j + nvertices * i];
			    
			    if (!valid)
				tokill.push_back(i);
			}

			ctcscoll->remove(&tokill.front(), tokill.size());
			ctcscoll->clear_velocity();
		    }

		    if (pushtheflow)
			driving_acceleration = hydrostatic_a;

		    CUDA_CHECK(cudaPeekAtLastError());

		    if (rank == 0)
			printf("the simulation begins now and it consists of %.3e steps\n", (double)(nsteps - it));

		    time_simulation_start = MPI_Wtime();
		}
		
		tstart = MPI_Wtime();
		cells.build(particles.xyzuvw.data, particles.size, mainstream);
		timings["build-cells"] += MPI_Wtime() - tstart;
		
		CUDA_CHECK(cudaPeekAtLastError());
		
		//THIS IS WHERE WE WANT TO ACHIEVE 70% OF THE PEAK
		//TODO: i need a coordinating class that performs all the local work while waiting for the communication
		{
		    tstart = MPI_Wtime();
		    
		    dpd.pack(particles.xyzuvw.data, particles.size, cells.start, cells.count, mainstream);
		    dpd.local_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, mainstream);
		    
		    if (wall)
			wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, 
					   cells.start, cells.count, mainstream);

		    dpd.consolidate_and_post(particles.xyzuvw.data, particles.size, mainstream);
		    dpd.wait_for_messages(mainstream);
		    dpd.remote_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, mainstream);

		    timings["evaluate-interactions"] += MPI_Wtime() - tstart; 
		    
		    CUDA_CHECK(cudaPeekAtLastError());	
		    	
		    if (rbcscoll)
		    {
			tstart = MPI_Wtime();
			rbc_interactions.evaluate(particles.xyzuvw.data, particles.size, particles.axayaz.data,
						  cells.start, cells.count, rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);
			timings["evaluate-rbc"] += MPI_Wtime() - tstart;
		    }
		    
		    CUDA_CHECK(cudaPeekAtLastError());

		    if (ctcscoll)
		    {
			tstart = MPI_Wtime();
			ctc_interactions.evaluate(particles.xyzuvw.data, particles.size, particles.axayaz.data,
						  cells.start, cells.count, ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);
			timings["evaluate-ctc"] += MPI_Wtime() - tstart;
		    }
		    
		    CUDA_CHECK(cudaPeekAtLastError());

		    if (wall)
		    {
			tstart = MPI_Wtime();
		
			if (rbcscoll)
			    wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

			if (ctcscoll)
			    wall->interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, mainstream);

			timings["body-walls interactions"] += MPI_Wtime() - tstart;
		    }
		}
	
		CUDA_CHECK(cudaPeekAtLastError());
	
		if (it % steps_per_dump == 0)
		{
		    NVTX_RANGE("data-dump");
		    CUDA_CHECK(cudaStreamSynchronize(mainstream));

		    tstart = MPI_Wtime();
		    int n = particles.size;

		    if (rbcscoll)
			n += rbcscoll->pcount();

		    if (ctcscoll)
			n += ctcscoll->pcount();

		    Particle * p = new Particle[n];
		    Acceleration * a = new Acceleration[n];

		    CUDA_CHECK(cudaMemcpy(p, particles.xyzuvw.data, sizeof(Particle) * particles.size, cudaMemcpyDeviceToHost));
		    CUDA_CHECK(cudaMemcpy(a, particles.axayaz.data, sizeof(Acceleration) * particles.size, cudaMemcpyDeviceToHost));
		   
		    int start = particles.size;

		    if (rbcscoll)
		    {
			CUDA_CHECK(cudaMemcpy(p + start, rbcscoll->xyzuvw.data, sizeof(Particle) * rbcscoll->pcount(), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(a + start, rbcscoll->axayaz.data, sizeof(Acceleration) * rbcscoll->pcount(), cudaMemcpyDeviceToHost));

			start += rbcscoll->pcount();
		    }

		    if (ctcscoll)
		    {
			CUDA_CHECK(cudaMemcpy(p + start, ctcscoll->xyzuvw.data, sizeof(Particle) * ctcscoll->pcount(), cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaMemcpy(a + start, ctcscoll->axayaz.data, sizeof(Acceleration) * ctcscoll->pcount(), cudaMemcpyDeviceToHost));

			start += ctcscoll->pcount();
		    }

		    assert(start == n);

		    diagnostics(activecomm, cartcomm, p, n, dt, it, a);
		    
		    if (xyz_dumps)
			xyz_dump(activecomm, cartcomm, "particles.xyz", "all-particles", p, n, it > 0);
		  
		    if (hdf5part_dumps)
			if (dump_part_solvent)
			    dump_part_solvent->dump(p, n);
			else
			    dump_part.dump(p, n);

		    if (hdf5field_dumps)
			dump_field.dump(activecomm, p, particles.size, it);

		    if (rbcscoll)
			rbcscoll->dump(activecomm, cartcomm);
		   	
		    if (ctcscoll)
			ctcscoll->dump(activecomm, cartcomm);

		    delete [] p;
		    delete [] a;

		    timings["diagnostics"] += MPI_Wtime() - tstart;
		}

		tstart = MPI_Wtime();
		particles.update_stage2_and_1(driving_acceleration, mainstream);

		CUDA_CHECK(cudaPeekAtLastError());

		if (rbcscoll)
		    rbcscoll->update_stage2_and_1(driving_acceleration, mainstream);

		CUDA_CHECK(cudaPeekAtLastError());

		if (ctcscoll)
		    ctcscoll->update_stage2_and_1(driving_acceleration, mainstream);
		timings["update"] += MPI_Wtime() - tstart;
		
		if (wall)
		{
		    tstart = MPI_Wtime();
		    wall->bounce(particles.xyzuvw.data, particles.size, mainstream);
		    
		    if (rbcscoll)
			wall->bounce(rbcscoll->data(), rbcscoll->pcount(), mainstream);

		    if (ctcscoll)
			wall->bounce(ctcscoll->data(), ctcscoll->pcount(), mainstream);

		    timings["bounce-walls"] += MPI_Wtime() - tstart;
		}

		CUDA_CHECK(cudaPeekAtLastError());
	    }

	    const double time_simulation_stop = MPI_Wtime();
	    const double telapsed = time_simulation_stop - time_simulation_start;

	    if (rank == 0)
		if (it == nsteps)
		    printf("simulation is done after %.3e s (%dm%ds). Ciao.\n", 
			   telapsed, (int)(telapsed / 60), (int)(telapsed) % 60);
		else
		    if (it != wall_creation_stepid)
			printf("external termination request (signal) after %.3e s. Bye.\n", telapsed);
	    
	    fflush(stdout);
	    
	    CUDA_CHECK(cudaStreamDestroy(mainstream));
	
	    if (wall)
		delete wall;

	    if (rbcscoll)
		delete rbcscoll;

	    if (ctcscoll)
		delete ctcscoll;

	    if (dump_part_solvent)
		delete dump_part_solvent;
	}
	
	if (activecomm != cartcomm)
	    MPI_CHECK(MPI_Comm_free(&activecomm));
   
	MPI_CHECK(MPI_Comm_free(&cartcomm));
	
	MPI_CHECK( MPI_Finalize() );
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
