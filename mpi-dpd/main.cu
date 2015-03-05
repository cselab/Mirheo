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
    
    int nranks, rank;   
    
    {
	MPI_CHECK( MPI_Init(&argc, &argv) );

	MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	
	srand48(rank);
	
	MPI_Comm cartcomm;
	int periods[] = {1, 1, 1};	    
	MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );
	
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

	    H5PartDump dump_part("allparticles.h5part", cartcomm);
	    H5FieldDump dump_field(cartcomm);

	    RedistributeParticles redistribute(cartcomm);
	    RedistributeRBCs redistribute_rbcs(cartcomm);
	    RedistributeCTCs redistribute_ctcs(cartcomm);

	    ComputeInteractionsDPD dpd(cartcomm);
	    ComputeInteractionsRBC rbc_interactions(cartcomm);
	    ComputeInteractionsCTC ctc_interactions(cartcomm);
	    ComputeInteractionsWall * wall = NULL;
	    
	    cudaStream_t mainstream;
	    CUDA_CHECK(cudaStreamCreate(&mainstream));
	    	    
	    CUDA_CHECK(cudaPeekAtLastError());

	    cells.build(particles.xyzuvw.data, particles.size, mainstream);

	    dpd.evaluate(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, mainstream);
    
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

	    const size_t nsteps = (int)(tend / dt);
	    int it;

	    for(it = 0; it < nsteps; ++it)
	    {
		//if (it > 499)printf("it is %d\n", it);

#ifdef _USE_NVTX_
		if (it == 7000)
		{
		    currently_profiling = true;
		    CUDA_CHECK(cudaProfilerStart());
		    
		}
		else if (it == 7050)
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
			MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &exitrequest, 1, MPI_LOGICAL, MPI_LOR, cartcomm)); 
			graceful_exit = exitrequest;
		    }

		    if (graceful_exit)
		    {
			if (!rank)
			    printf("Got a termination request. Time to save and exit!\n");
		    
			break;
		    }	    		    

		    report_host_memory_usage(cartcomm, stdout);
		    
		    if (rank == 0)
		    {
			static double t0 = MPI_Wtime(), t1;

			t1 = MPI_Wtime();
		    
			if (it > 0)
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
		const int newnp = redistribute.stage1(particles.xyzuvw.data, particles.size, mainstream);
		particles.resize(newnp);
		redistribute.stage2(particles.xyzuvw.data, particles.size, mainstream);
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
		if (walls && it > 5000 && wall == NULL)
		{
		    CUDA_CHECK(cudaDeviceSynchronize());

		    int nsurvived = 0;
		    ExpectedMessageSizes new_sizes;
		    wall = new ComputeInteractionsWall(cartcomm, particles.xyzuvw.data, particles.size, nsurvived, new_sizes);
		    
		    redistribute.adjust_message_sizes(new_sizes);
		    dpd.adjust_message_sizes(new_sizes);

		    particles.resize(nsurvived);
		    particles.clear_velocity();
		    		    
		    if (rank == 0)
		    {
			if( access( "trajectories.xyz", F_OK ) != -1 )
			{
			    const int retval = rename ("trajectories.xyz", "trajectories-equilibration.xyz");
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
		}
		
		tstart = MPI_Wtime();
		cells.build(particles.xyzuvw.data, particles.size, mainstream);
		timings["build-cells"] += MPI_Wtime() - tstart;
		
		CUDA_CHECK(cudaPeekAtLastError());
		
		//THIS IS WHERE WE WANT TO ACHIEVE 70% OF THE PEAK
		//TODO: i need a coordinating class that performs all the local work while waiting for the communication
		{
		    tstart = MPI_Wtime();
		    dpd.evaluate(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, mainstream);
		    timings["evaluate-dpd"] += MPI_Wtime() - tstart;
		    
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
			wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, 
					   cells.start, cells.count, mainstream);

			if (rbcscoll)
			    wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

			if (ctcscoll)
			    wall->interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, mainstream);

			timings["evaluate-walls"] += MPI_Wtime() - tstart;
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

		    diagnostics(cartcomm, p, n, dt, it, a);
		    
		    if (xyz_dumps)
			xyz_dump(cartcomm, "particles.xyz", "all-particles", p, n, it > 0);
		  
		    if (hdf5part_dumps)
			dump_part.dump(p, n);

		    if (hdf5field_dumps)
			dump_field.dump(p, n, it);

		    if (rbcscoll && it % steps_per_dump == 0)
			rbcscoll->dump(cartcomm);
		   	
		    if (ctcscoll && it % steps_per_dump == 0)
			ctcscoll->dump(cartcomm);

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

	    if (rank == 0)
		if (it == nsteps)
		    printf("simulation is done. Ciao.\n");
		else
		    printf("external termination request (signal). Bye.\n");

	    fflush(stdout);
	    
	    CUDA_CHECK(cudaStreamDestroy(mainstream));
	
	    if (wall)
		delete wall;

	    if (rbcscoll)
		delete rbcscoll;

	    if (ctcscoll)
		delete ctcscoll;
	}
	   
	MPI_CHECK(MPI_Comm_free(&cartcomm));
 
	MPI_CHECK( MPI_Finalize() );
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
