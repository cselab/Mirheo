#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <sys/stat.h>
#include <mpi.h>

#include <vector>
#include <map>

#include "common.h"
#include "containers.h"
#include "dpd-interactions.h"
#include "wall-interactions.h"
#include "redistribute-particles.h"
#include "redistribute-rbcs.h"
#include "rbc-interactions.h"

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

    CUDA_CHECK(cudaSetDevice(0));

    int nranks, rank;   
    
    {
	MPI_CHECK( MPI_Init(&argc, &argv) );
    
	{
	    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

	    srand48(rank);
	    
	    MPI_Comm cartcomm;
	    int periods[] = {1, 1, 1};	    
	    MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );
	
	    vector<Particle> ic(L * L * L * 3  );
	    
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
		rbcscoll = new CollectionRBC(L);
	    
	    RedistributeParticles redistribute(cartcomm, L);
	    RedistributeRBCs redistribute_rbcs(cartcomm, L);

	    ComputeInteractionsDPD dpd(cartcomm, L);
	    ComputeInteractionsRBC rbc_interactions(cartcomm, L);
	    ComputeInteractionsWall * wall = NULL;
	    
	    cudaStream_t stream;
	    CUDA_CHECK(cudaStreamCreate(&stream));
	    	    
	    redistribute_rbcs.stream = stream;

	    int saru_tag = rank;
	    
	    cells.build(particles.xyzuvw.data, particles.size);
	    std::map<string, double> timings;
	    dpd.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, timings);
	    
	    if (rbcscoll)
		rbc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
					  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc());

	    float dpdx[3] = {0, 0, 0};
		    
	    const size_t nsteps = (int)(tend / dt);

	    for(int it = 0; it < nsteps; ++it)
	    {
		if (it % steps_per_report == 0)
		{
		    report_host_memory_usage(cartcomm, stdout);

		    if (rank == 0)
		    {
			static double t0 = MPI_Wtime(), t1;

			t1 = MPI_Wtime();
		    
			if (it > 0)
			{
			    printf("beginning of time step %d (%.3e s)\n", it, t1 - t0);
			    printf("in more details:\n");
			    double tt = 0;
			    for(std::map<string, double>::iterator it = timings.begin(); it != timings.end(); ++it)
			    {
				printf("%s: %.3e s\n", it->first.c_str(), it->second);
				tt += it->second;
				it->second = 0;
			    }
			    printf("discrepancy: %.3e s\n", (t1 - t0) - tt);
			}

			t0 = t1;
		    }
		}
	    
		double tstart;

		if (it == 0)
		{
		    particles.update_stage1(dpdx);
		    
		    if (rbcscoll)
			rbcscoll->update_stage1();
		}

		tstart = MPI_Wtime();
		const int newnp = redistribute.stage1(particles.xyzuvw.data, particles.size);
		particles.resize(newnp);
		redistribute.stage2(particles.xyzuvw.data, particles.size);
		timings["redistribute-particles"] += MPI_Wtime() - tstart;

		if (rbcscoll)
		{	
		    tstart = MPI_Wtime();
		    const int nrbcs = redistribute_rbcs.stage1(rbcscoll->data(), rbcscoll->count());
		    rbcscoll->resize(nrbcs);
		    redistribute_rbcs.stage2(rbcscoll->data(), rbcscoll->count());
		    timings["redistribute-rbc"] += MPI_Wtime() - tstart;
		}

		//create the wall when it is time
		if (walls && it > 500 && wall == NULL)
		{
		    int nsurvived = 0;
		    wall = new ComputeInteractionsWall(cartcomm, L, particles.xyzuvw.data, particles.size, nsurvived);
		    
		    particles.resize(nsurvived);
		    		    
		    if (rank == 0)
		    {
			if( access( "trajectories.xyz", F_OK ) != -1 )
			{
			    const int retval = rename ("trajectories.xyz", "trajectories-equilibration.xyz");
			    assert(retval != -1);
			}
		    
			if( access( "rbcscoll.xyz", F_OK ) != -1 )
			{
			    const int retval = rename ("rbcscoll.xyz", "rbcscoll-equilibration.xyz");
			    assert(retval != -1);
			}
		    }

		    //remove Rbcscoll touching the wall
		    if(rbcscoll)
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
		    }

		    if (pushtheflow)
			dpdx[0] = -0.1;
		}

		tstart = MPI_Wtime();
		cells.build(particles.xyzuvw.data, particles.size);
		timings["build-cells"] += MPI_Wtime() - tstart;

		//THIS IS WHERE WE WANT TO ACHIEVE 70% OF THE PEAK
		//TODO: i need a coordinating class that performs all the local work while waiting for the communication
		{
		    //tstart = MPI_Wtime();
		    dpd.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, timings);
		    //timings["evaluate-dpd"] += MPI_Wtime() - tstart;

		    if (rbcscoll)
		    {
			tstart = MPI_Wtime();
			rbc_interactions.evaluate(saru_tag, particles.xyzuvw.data, particles.size, particles.axayaz.data,
						  cells.start, cells.count, rbcscoll->data(), rbcscoll->count(), rbcscoll->acc());
			timings["evaluate-rbc"] += MPI_Wtime() - tstart;
		    }

		    if (wall)
		    {
			tstart = MPI_Wtime();
			wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, 
					   cells.start, cells.count, saru_tag);

			if (rbcscoll)
			    wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, saru_tag);

			timings["evaluate-walls"] += MPI_Wtime() - tstart;
		    }
		}
		
		particles.update_stage2_and_1(dpdx);

		if (rbcscoll)
		    rbcscoll->update_stage2_and_1();

		if (wall)
		{
		    tstart = MPI_Wtime();
		    wall->bounce(particles.xyzuvw.data, particles.size);
		    
		    if (rbcscoll)
			wall->bounce(rbcscoll->data(), rbcscoll->pcount());
		    timings["bounce-walls"] += MPI_Wtime() - tstart;
		}
	    
		if (it % steps_per_report == 0)
		{
		    const int n = particles.size;

		    Particle * p = new Particle[n];
		    Acceleration * a = new Acceleration[n];

		    CUDA_CHECK(cudaMemcpy(p, particles.xyzuvw.data, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
		    CUDA_CHECK(cudaMemcpy(a, particles.axayaz.data, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
		   
		    //we fused VV stages so we need to recover the state before stage 1
		    for(int i = 0; i < n; ++i)
			for(int c = 0; c < 3; ++c)
			{
			    assert(!isnan(p[i].x[c]));
			    assert(!isnan(p[i].u[c]));
			    assert(!isnan(a[i].a[c]));
	    
			    p[i].x[c] -= dt * p[i].u[c];
			    p[i].u[c] -= 0.5 * dt * a[i].a[c];
			}

		    diagnostics(cartcomm, p, n, dt, it, L, a);
		    
		    if (rbcscoll)
			rbcscoll->dump(cartcomm);
		   
		    delete [] p;
		    delete [] a;
		}
	    }

	    CUDA_CHECK(cudaStreamDestroy(stream));
	
	    if (wall)
		delete wall;

	    if (rbcscoll)
		delete rbcscoll;

	    MPI_CHECK(MPI_Comm_free(&cartcomm));
	}
	
	MPI_CHECK( MPI_Finalize() );
	
	if (rank == 0)
	    printf("simulation is done. Ciao.\n");
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
	
