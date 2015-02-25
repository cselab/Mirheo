#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <vector>
#include <map>

#include "common.h"
#include "io.h"
#include "containers.h"
#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "rbc-interactions.h"

#include "ctc.h"
#include "Rank2Node.c"

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
    
    double tevaldpd=0.;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int nranks, rank;   
    {
	MPI_CHECK( MPI_Init(&argc, &argv) );
    	static double maxtime=MPI_Wtime(); 

	MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	
	srand48(rank);
	
	MPI_Comm cartcomm;
	MPI_Comm AROcomm;
	int periods[] = {1, 1, 1};	    
#if 1
	MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );
#else
#define RANK2NODE "rank2node.dat"
        int *aranks=new int[nranks];
        FILE *fp=fopen(RANK2NODE,"r");
        if(fp==NULL) {
           fprintf(stderr,"Error opening %s file\n",RANK2NODE);
           MPI_Abort(MPI_COMM_WORLD,2);
           exit(1);
        }
        int countr=0;
        {
#define MAXINPUTLINE 1024
#define DELIMITER " \t"
         char line[MAXINPUTLINE];
         char *token;

         bool survive=false;
         while (fgets(line, MAXINPUTLINE, fp)) {
               token=strtok(line,DELIMITER);
               while(token!=NULL) {
                  aranks[countr]=atoi(token);
                  if(aranks[countr]==rank) {
                     survive=true;
                  }
                  countr++;
                  token=strtok(NULL,DELIMITER);
               }
          }
          fclose(fp);
#if 0
          if(rank==0) {
           for(int i=0; i<countr; i++) {
               printf("rank %d survive!\n",aranks[i]);
           }
          }
          if(!survive) {
             printf("rank %d bye bye\n",rank);
          }
#endif
        }
        MPI_Group orig_group, new_group;
        MPI_CHECK( MPI_Comm_group(MPI_COMM_WORLD, &orig_group) );
        MPI_CHECK( MPI_Group_incl(orig_group, countr, aranks, &new_group) );
        MPI_CHECK( MPI_Comm_create(MPI_COMM_WORLD, new_group, &AROcomm) );
        MPI_CHECK(MPI_Comm_dup(AROcomm,&cartcomm));

        MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartcomm) );
#endif
        int mylorank;
        Rank2Node(&mylorank,cartcomm);
	
	{
	    vector<Particle> ic(L * L * L * 3);
	    
	    for(int i = 0; i < ic.size(); ++i)
		for(int c = 0; c < 3; ++c)
		{
		    ic[i].x[c] = -L * 0.5 + drand48() * L;
		    ic[i].u[c] = 0;
		}
	    	    	  
	    ParticleArray particles(ic);
	    CellLists cells(L);		  
	    CollectionRBC * rbcscoll = NULL;
	    
	    if (rbcs)
	    {
		rbcscoll = new CollectionRBC(cartcomm, L);
		rbcscoll->setup();
	    }

	    CollectionCTC * ctcscoll = NULL;

	    if (ctcs)
	    {
		ctcscoll = new CollectionCTC(cartcomm, L);
		ctcscoll->setup();
	    }

	    H5PartDump dump_part("allparticles.h5part", cartcomm, L);
	    H5FieldDump dump_field(cartcomm);

	    RedistributeParticles redistribute(cartcomm, L);
	    RedistributeRBCs redistribute_rbcs(cartcomm, L);
	    RedistributeCTCs redistribute_ctcs(cartcomm, L);

	    ComputeInteractionsDPD dpd(cartcomm, L);
	    ComputeInteractionsRBC rbc_interactions(cartcomm, L);
	    ComputeInteractionsCTC ctc_interactions(cartcomm, L);
	    ComputeInteractionsWall * wall = NULL;
	    
	    cudaStream_t stream;
	    CUDA_CHECK(cudaStreamCreate(&stream));
	    	    
	    redistribute_rbcs.stream = stream;

	    int saru_tag = rank;
	    
	    CUDA_CHECK(cudaPeekAtLastError());

	    cells.build(particles.xyzuvw.data, particles.size);

	    dpd.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count);
    
	    if (rbcscoll)
		rbc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
					  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc());

	    if (ctcscoll)
		ctc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
					ctcscoll->data(), ctcscoll->count(), ctcscoll->acc());

	    float driving_acceleration = 0;

	    if (!walls && pushtheflow)
		driving_acceleration = hydrostatic_a;		    

	    std::map<string, double> timings;

	    const size_t nsteps = (int)(tend / dt);
	    int it;

	    for(it = 0; it < nsteps; ++it)
	    {
		if (it % steps_per_report == 0)
		{ 
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
			    printf("beginning of time step %d (%.3f ms)\n", it, (t1 - t0) * 1e3 / steps_per_report);
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
		    particles.update_stage1(driving_acceleration);
		    
		    if (rbcscoll)
			rbcscoll->update_stage1(driving_acceleration);

		    if (ctcscoll)
			ctcscoll->update_stage1(driving_acceleration);
		}

		tstart = MPI_Wtime();
		const int newnp = redistribute.stage1(particles.xyzuvw.data, particles.size);
		particles.resize(newnp);
		redistribute.stage2(particles.xyzuvw.data, particles.size);
		timings["redistribute-particles"] += MPI_Wtime() - tstart;
		
		CUDA_CHECK(cudaPeekAtLastError());

		if (rbcscoll)
		{	
		    tstart = MPI_Wtime();
		    const int nrbcs = redistribute_rbcs.stage1(rbcscoll->data(), rbcscoll->count());
		    rbcscoll->resize(nrbcs);
		    redistribute_rbcs.stage2(rbcscoll->data(), rbcscoll->count());
		    timings["redistribute-rbc"] += MPI_Wtime() - tstart;
		}

		CUDA_CHECK(cudaPeekAtLastError());
			
		if (ctcscoll)
		{	
		    tstart = MPI_Wtime();
		    const int nctcs = redistribute_ctcs.stage1(ctcscoll->data(), ctcscoll->count());
		    ctcscoll->resize(nctcs);
		    redistribute_ctcs.stage2(ctcscoll->data(), ctcscoll->count());
		    timings["redistribute-ctc"] += MPI_Wtime() - tstart;
		}

		CUDA_CHECK(cudaPeekAtLastError());
		
		tstart = MPI_Wtime();
		CUDA_CHECK(cudaStreamSynchronize(redistribute.mystream));
		CUDA_CHECK(cudaStreamSynchronize(redistribute_rbcs.stream));
		timings["stream-synchronize"] += MPI_Wtime() - tstart;
		
		//create the wall when it is time
		if (walls && it > 5000 && wall == NULL)
		{
		    int nsurvived = 0;
		    wall = new ComputeInteractionsWall(cartcomm, L, particles.xyzuvw.data, particles.size, nsurvived);
		    
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
			
			SolidWallsKernel::fill_keys<<< (rbcscoll->pcount() + 127) / 128, 128 >>>(rbcscoll->data(), rbcscoll->pcount(), L, marks.data);
			
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
			
			SolidWallsKernel::fill_keys<<< (ctcscoll->pcount() + 127) / 128, 128 >>>(ctcscoll->data(), ctcscoll->pcount(), L, 
												 marks.data);
			
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
		cells.build(particles.xyzuvw.data, particles.size);
		timings["build-cells"] += MPI_Wtime() - tstart;
		
		CUDA_CHECK(cudaPeekAtLastError());
		
		//THIS IS WHERE WE WANT TO ACHIEVE 70% OF THE PEAK
		//TODO: i need a coordinating class that performs all the local work while waiting for the communication
		{
		    tstart = MPI_Wtime();
		    if((it % steps_per_report) == 0) {
			cudaEventRecord(start);
		    }
		    dpd.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count);
		    if((it % steps_per_report) == 0) {
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			float *rbuftime=new float[nranks];		
		  	MPI_CHECK ( MPI_Gather(&milliseconds,1,MPI_FLOAT,rbuftime, 1,
                         MPI_FLOAT, 0, MPI_COMM_WORLD) );
			if(rank==0) {
			 printf("Timing dpd.evaluate it %d\n",it);
			 for(int i=0; i<nranks; i++) {
		             printf("%7.5f ",rbuftime[i]);
	                 }
			 printf("\n End timing dpd.evaluate\n");
		         delete [] rbuftime;
		        }
		    	tevaldpd+=milliseconds;
		    }
		    timings["evaluate-dpd"] += (MPI_Wtime() - tstart);
		    
		    CUDA_CHECK(cudaPeekAtLastError());	
		    	
		    if (rbcscoll)
		    {
			tstart = MPI_Wtime();
			rbc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data,
						  cells.start, cells.count, rbcscoll->data(), rbcscoll->count(), rbcscoll->acc());
			timings["evaluate-rbc"] += MPI_Wtime() - tstart;
		    }
		    
		    CUDA_CHECK(cudaPeekAtLastError());

		    if (ctcscoll)
		    {
			tstart = MPI_Wtime();
			ctc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data,
						  cells.start, cells.count, ctcscoll->data(), ctcscoll->count(), ctcscoll->acc());
			timings["evaluate-ctc"] += MPI_Wtime() - tstart;
		    }
		    
		    CUDA_CHECK(cudaPeekAtLastError());

		    if (wall)
		    {
			tstart = MPI_Wtime();
			wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, 
					   cells.start, cells.count, saru_tag);

			if (rbcscoll)
			    wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, saru_tag);

			if (ctcscoll)
			    wall->interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, saru_tag);

			timings["evaluate-walls"] += MPI_Wtime() - tstart;
		    }

		    //CUDA_CHECK(cudaDeviceSynchronize());
		}
		
		CUDA_CHECK(cudaPeekAtLastError());

		if (it % steps_per_dump == 0)
		{
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

		    diagnostics(cartcomm, p, n, dt, it, L, a);
		    
		    if (xyz_dumps)
			xyz_dump(cartcomm, "particles.xyz", "all-particles", p, n, L, it > 0);
		  
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
		particles.update_stage2_and_1(driving_acceleration);

		CUDA_CHECK(cudaPeekAtLastError());

		if (rbcscoll)
		    rbcscoll->update_stage2_and_1(driving_acceleration);

		CUDA_CHECK(cudaPeekAtLastError());

		if (ctcscoll)
		    ctcscoll->update_stage2_and_1(driving_acceleration);
		timings["update"] += MPI_Wtime() - tstart;
		
		if (wall)
		{
		    tstart = MPI_Wtime();
		    wall->bounce(particles.xyzuvw.data, particles.size);
		    
		    if (rbcscoll)
			wall->bounce(rbcscoll->data(), rbcscoll->pcount());

		    if (ctcscoll)
			wall->bounce(ctcscoll->data(), ctcscoll->pcount());

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
	    
	    CUDA_CHECK(cudaStreamDestroy(stream));
	
	    if (wall)
		delete wall;

	    if (rbcscoll)
		delete rbcscoll;

	    if (ctcscoll)
		delete ctcscoll;
	}
	   
	MPI_CHECK(MPI_Comm_free(&cartcomm));
	{
        double *rbuftime=new double[nranks];	
	MPI_CHECK ( MPI_Gather(&tevaldpd,1,MPI_DOUBLE,rbuftime, 1,
                         MPI_DOUBLE, 0, MPI_COMM_WORLD) );
	if(rank==0) {
//	 for(int i=0; i<nranks; i++) {
//		printf("rank %d: time in dpd.evaluate: %f\n",i,rbuftime[i]);
//	 }
#define RANK2NODE "MPICH_RANK_ORDER"
#define MAXINPUTLINE 1024
#define DELIMITER ","
         char line[MAXINPUTLINE];
         char *token;
         int *aranks=new int[nranks];
         FILE *fp=fopen(RANK2NODE,"r");
         if(fp==NULL) {
           fprintf(stderr,"Error opening %s file\n",RANK2NODE);
           MPI_Abort(MPI_COMM_WORLD,2);
           exit(1);
         }
	 int countr, j=0;
         while (fgets(line, MAXINPUTLINE, fp)) {
               countr=0;
               token=strtok(line,DELIMITER);
               while(token!=NULL) {
                  aranks[countr]=atoi(token);
                  if(aranks[countr]==rank) {
                  }
                  countr++;
                  token=strtok(NULL,DELIMITER);
               }
	       double timexnode=0;
	       for(int i=0; i<countr; i++) {
			timexnode+=rbuftime[aranks[i]];		
	       }
	       printf("Total time in dpd.evaluate for node %d: %f\n",j,timexnode);
	       j++;
          }
          fclose(fp);
	  delete [] aranks;
	}
	delete [] rbuftime;
	}
    	printf("Task %d: time=%f\n",rank,MPI_Wtime()-maxtime);
	MPI_CHECK( MPI_Finalize() );
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop); 
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
