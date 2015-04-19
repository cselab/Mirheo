/*
 *  simulation.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-24.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include "simulation.h"

std::vector<Particle> Simulation::_ic()
{
    srand48(rank);

    std::vector<Particle> ic(XSIZE_SUBDOMAIN * YSIZE_SUBDOMAIN * ZSIZE_SUBDOMAIN * numberdensity);
    
    const int L[3] = { XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN };
    
    for(int iz = 0; iz < L[2]; iz++)
	for(int iy = 0; iy < L[1]; iy++)
	    for(int ix = 0; ix < L[0]; ix++)
		for(int l = 0; l < numberdensity; ++l)
		{
		    const int p = l + numberdensity * (ix + L[0] * (iy + L[1] * iz));
		    
		    ic[p].x[0] = -L[0]/2 + ix + drand48();
		    ic[p].x[1] = -L[1]/2 + iy + drand48();
		    ic[p].x[2] = -L[2]/2 + iz + drand48();
		    ic[p].u[0] = 0;
		    ic[p].u[1] = 0;
		    ic[p].u[2] = 0;
		}

    /* use this to check robustness 
    for(int i = 0; i < ic.size(); ++i)
	for(int c = 0; c < 3; ++c)
	    {
		ic[i].x[c] = -L[c] * 0.5 + drand48() * L[c];
		ic[i].u[c] = 0;
	    }
    */
    
    return ic;
}

void Simulation::_redistribute()
{
    double tstart = MPI_Wtime();
    
    redistribute.pack(particles.xyzuvw.data, particles.size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll) 
	redistribute_rbcs.extent(rbcscoll->data(), rbcscoll->count(), mainstream);
    
    if (ctcscoll)
	redistribute_ctcs.extent(ctcscoll->data(), ctcscoll->count(), mainstream);
    
    redistribute.send();

    if (rbcscoll) 
	redistribute_rbcs.pack_sendcount(rbcscoll->data(), rbcscoll->count(), mainstream);
    
    if (ctcscoll)
	redistribute_ctcs.pack_sendcount(ctcscoll->data(), ctcscoll->count(), mainstream);
    
    redistribute.bulk(particles.size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    const int newnp = redistribute.recv_count(mainstream, host_idle_time);
    
    int nrbcs;
    if (rbcscoll) 
	nrbcs = redistribute_rbcs.post();

    int nctcs;
    if (ctcscoll)
	nctcs = redistribute_ctcs.post();

    if (rbcscoll) 
	rbcscoll->resize(nrbcs);

    if (ctcscoll)
	ctcscoll->resize(nctcs);

    unordered_particles.resize(newnp);

    redistribute.recv_unpack(unordered_particles.data, newnp, mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    particles.resize(newnp);

    cells.build(particles.xyzuvw.data, particles.size, mainstream, NULL, unordered_particles.data);
    
    if (rbcscoll)
	redistribute_rbcs.unpack(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.unpack(ctcscoll->data(), ctcscoll->count(), mainstream);
    
    CUDA_CHECK(cudaPeekAtLastError());

    localcomm.barrier();

    timings["redistribute"] += MPI_Wtime() - tstart;
}

void Simulation::_report(const bool verbose, const int idtimestep)
{ 
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
	
	if (verbose && imbalance >= 0)
	    printf("\x1b[93moverall imbalance: %.f%%, host workload min/avg/max: %.2f/%.2f/%.2f ms\x1b[0m\n", 
		   imbalance , minval, sumval / commsize, maxval);
	
	host_idle_time = 0;
	t0 = t1;
    }
    
    {
	static double t0 = MPI_Wtime(), t1;
	
	t1 = MPI_Wtime();
	
	if (verbose)
	{
	    printf("\x1b[92mbeginning of time step %d (%.3f ms)\x1b[0m\n", idtimestep, (t1 - t0) * 1e3 / steps_per_report);
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

void Simulation::_remove_bodies_from_wall(CollectionRBC * coll)
{
    if(!coll || !coll->count())
	return;
    
    SimpleDeviceBuffer<int> marks(coll->pcount());
    
    SolidWallsKernel::fill_keys<<< (coll->pcount() + 127) / 128, 128 >>>(coll->data(), coll->pcount(), marks.data);
    
    vector<int> tmp(marks.size);
    CUDA_CHECK(cudaMemcpy(tmp.data(), marks.data, sizeof(int) * marks.size, cudaMemcpyDeviceToHost));
    
    const int nbodies = coll->count();
    const int nvertices = coll->nvertices;
    
    std::vector<int> tokill;
    for(int i = 0; i < nbodies; ++i)
    {
	bool valid = true;
	
	for(int j = 0; j < nvertices && valid; ++j)
	    valid &= 0 == tmp[j + nvertices * i];
	
	if (!valid)
	    tokill.push_back(i);
    }
    
    coll->remove(&tokill.front(), tokill.size());
    coll->clear_velocity();
    
    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_create_walls(const bool verbose, bool & termination_request)
{
    if (verbose)
	printf("creation of the walls...\n");
    
    int nsurvived = 0;
    ExpectedMessageSizes new_sizes;
    wall = new ComputeInteractionsWall(cartcomm, particles.xyzuvw.data, particles.size, nsurvived, new_sizes, verbose);
    
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
	    
	    termination_request = true;
	    return;
	}
    }
    
    if (hdf5part_dumps)
	dump_part_solvent = new H5PartDump("solvent-particles.h5part", activecomm, cartcomm);
    
    particles.resize(nsurvived);
    particles.clear_velocity();

    CUDA_CHECK(cudaPeekAtLastError());    
    
    //remove cells touching the wall
    _remove_bodies_from_wall(rbcscoll);
    _remove_bodies_from_wall(ctcscoll);

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
}

void Simulation::_forces()
{    
    double tstart = MPI_Wtime();
	
    particles.clear_acc(mainstream);

    if (rbcscoll)
	rbcscoll->clear_acc(mainstream);
    
    if (ctcscoll)
    	ctcscoll->clear_acc(mainstream);

    if (rbcscoll) 
	rbc_interactions.extent(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll) 
	ctc_interactions.extent(ctcscoll->data(), ctcscoll->count(), mainstream);
	
    if (rbcscoll) 
	rbc_interactions.count(rbcscoll->count());

    if (ctcscoll) 
	ctc_interactions.count(ctcscoll->count());

    dpd.pack(particles.xyzuvw.data, particles.size, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll) 
	rbc_interactions. pack_p(rbcscoll->data(), mainstream);

    if (ctcscoll) 
	ctc_interactions.pack_p(ctcscoll->data(), mainstream);

    dpd.consolidate_and_post(particles.xyzuvw.data, particles.size, mainstream);

    dpd.local_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll) 
	rbc_interactions.exchange_count();

    if (ctcscoll) 
	ctc_interactions.exchange_count();

    if (rbcscoll) 
	rbc_interactions.post_p();

    if (ctcscoll) 
	ctc_interactions.post_p();

    if (rbcscoll) 
	rbc_interactions.fsi_bulk(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);
	
    if (ctcscoll) 
	ctc_interactions.fsi_bulk(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				  ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);

    if (rbcscoll && wall)
	wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

    if (ctcscoll && wall)
	wall->interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, mainstream);

    if (rbcscoll) 
	rbc_interactions.fsi_halo(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, 
				  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);
	
    if (ctcscoll) 
	ctc_interactions.fsi_halo(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, 
				  ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);
	
    if (rbcscoll) 
	rbc_interactions.internal_forces(rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);

    if (ctcscoll) 
	ctc_interactions.internal_forces(ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);

    if (wall)
	wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, 
			   cells.start, cells.count, mainstream);
	
    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll) 
	rbc_interactions.post_a();

    if (ctcscoll) 
	ctc_interactions.post_a();

    
    dpd.wait_for_messages(mainstream);
    dpd.remote_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, mainstream);
	
    if (rbcscoll) 
	rbc_interactions.merge_a(rbcscoll->acc(), mainstream);

    if (ctcscoll) 
	ctc_interactions.merge_a(ctcscoll->acc(), mainstream);

    timings["interactions"] += MPI_Wtime() - tstart; 
	
    CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::_data_dump(const int idtimestep)
{
    NVTX_RANGE("data-dump");
    
    double tstart = MPI_Wtime();
    
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
    
    diagnostics(activecomm, cartcomm, p, n, dt, idtimestep, a);
	
    if (xyz_dumps)
	xyz_dump(activecomm, cartcomm, "particles.xyz", "all-particles", p, n, idtimestep > 0);
    
    if (hdf5part_dumps)
	if (dump_part_solvent)
	    dump_part_solvent->dump(p, n);
	else
	    dump_part.dump(p, n);
    
    if (hdf5field_dumps)
	dump_field.dump(activecomm, p, particles.size, idtimestep);
    
    if (rbcscoll)
	rbcscoll->dump(activecomm, cartcomm);
    
    if (ctcscoll)
	ctcscoll->dump(activecomm, cartcomm);
    
    delete [] p;
    delete [] a;
    
    timings["data-dump"] += MPI_Wtime() - tstart;
}

void Simulation::_update_and_bounce()
{
    double tstart = MPI_Wtime();
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

Simulation::Simulation(MPI_Comm cartcomm, MPI_Comm activecomm, bool (*check_termination)()) :  
    cartcomm(cartcomm), activecomm(activecomm),
    particles(_ic()), cells(XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN), 
    rbcscoll(NULL), ctcscoll(NULL), wall(NULL),
    redistribute(cartcomm),  redistribute_rbcs(cartcomm),  redistribute_ctcs(cartcomm),
    dpd(cartcomm), rbc_interactions(cartcomm), ctc_interactions(cartcomm),
    dump_part("allparticles.h5part", activecomm, cartcomm),  dump_field(cartcomm),  dump_part_solvent(NULL), 
    check_termination(check_termination),
    driving_acceleration(0), host_idle_time(0), nsteps((int)(tend / dt))
{
    //Side not of Yu-Hang:
    //in production runs replace the numbers with 4 unique ones that are same across ranks
    //KISS rng_trunk( 0x26F94D92, 0x383E7D1E, 0x46144B48, 0x6DDD73CB );
    
    localcomm.initialize(activecomm);

    MPI_CHECK( MPI_Comm_size(activecomm, &nranks) );
    MPI_CHECK( MPI_Comm_rank(activecomm, &rank) );
    
    CUDA_CHECK(cudaStreamCreate(&mainstream));
	
    if (rbcs)
    {
	rbcscoll = new CollectionRBC(cartcomm);
	rbcscoll->setup();
    }
    
    if (ctcs) 
    {
	ctcscoll = new CollectionCTC(cartcomm);
	ctcscoll->setup();
    }
}

void Simulation::_lockstep()
{
    double tstart = MPI_Wtime();

    particles.clear_acc(mainstream);

    if (rbcscoll)
	rbcscoll->clear_acc(mainstream);
    
    if (ctcscoll)
    	ctcscoll->clear_acc(mainstream);
    
    if (rbcscoll)
      rbc_interactions.extent(rbcscoll->data(), rbcscoll->count(), mainstream);
    
    if (ctcscoll)
	ctc_interactions.extent(ctcscoll->data(), ctcscoll->count(), mainstream);
   
    dpd.pack(particles.xyzuvw.data, particles.size, cells.start, cells.count, mainstream);

    if (rbcscoll)
	rbc_interactions.count(rbcscoll->count());

    if (ctcscoll)
	ctc_interactions.count(ctcscoll->count());

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	rbc_interactions.pack_p(rbcscoll->data(), mainstream);

    if (ctcscoll)
	ctc_interactions.pack_p(ctcscoll->data(), mainstream);

    dpd.consolidate_and_post(particles.xyzuvw.data, particles.size, mainstream);

    dpd.local_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    localcomm.barrier(); // peh: 1

    if (rbcscoll)
	rbc_interactions.exchange_count();

    if (ctcscoll)
	ctc_interactions.exchange_count();

    if (rbcscoll)
	rbc_interactions.post_p();

    if (ctcscoll)
	ctc_interactions.post_p();

    if (rbcscoll)
	rbc_interactions.fsi_bulk(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);

    if (ctcscoll)
	ctc_interactions.fsi_bulk(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				  ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);

    if (rbcscoll)
	rbc_interactions.fsi_halo(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				  rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);

    if (ctcscoll)
	ctc_interactions.fsi_halo(particles.xyzuvw.data, particles.size, particles.axayaz.data, cells.start, cells.count,
				  ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);

    if (rbcscoll)
	rbc_interactions.post_a();

    if (ctcscoll)
	ctc_interactions.post_a();

    if (wall)
	wall->interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data,
			   cells.start, cells.count, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    dpd.wait_for_messages(mainstream);

    dpd.remote_interactions(particles.xyzuvw.data, particles.size, particles.axayaz.data, mainstream);

    particles.update_stage2_and_1(driving_acceleration, mainstream);

    if (wall)
	wall->bounce(particles.xyzuvw.data, particles.size, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    redistribute.pack(particles.xyzuvw.data, particles.size, mainstream);

    redistribute.send();

    redistribute.bulk(particles.size, mainstream);

    if (rbcscoll)
	rbc_interactions.internal_forces(rbcscoll->data(), rbcscoll->count(), rbcscoll->acc(), mainstream);

    if (ctcscoll)
	ctc_interactions.internal_forces(ctcscoll->data(), ctcscoll->count(), ctcscoll->acc(), mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll && wall)
	wall->interactions(rbcscoll->data(), rbcscoll->pcount(), rbcscoll->acc(), NULL, NULL, mainstream);

    if (ctcscoll && wall)
	wall->interactions(ctcscoll->data(), ctcscoll->pcount(), ctcscoll->acc(), NULL, NULL, mainstream);

    if (rbcscoll)
	rbc_interactions.merge_a(rbcscoll->acc(), mainstream);

    if (ctcscoll)
	ctc_interactions.merge_a(ctcscoll->acc(), mainstream);

    if (rbcscoll)
	rbcscoll->update_stage2_and_1(driving_acceleration, mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

    if (ctcscoll)
	ctcscoll->update_stage2_and_1(driving_acceleration, mainstream);

    if (wall && rbcscoll)
	wall->bounce(rbcscoll->data(), rbcscoll->pcount(), mainstream);

    if (wall && ctcscoll)
	wall->bounce(ctcscoll->data(), ctcscoll->pcount(), mainstream);

    const int newnp = redistribute.recv_count(mainstream, host_idle_time);

    CUDA_CHECK(cudaPeekAtLastError());

    if (rbcscoll)
	redistribute_rbcs.extent(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.extent(ctcscoll->data(), ctcscoll->count(), mainstream);

    if (rbcscoll)
	redistribute_rbcs.pack_sendcount(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.pack_sendcount(ctcscoll->data(), ctcscoll->count(), mainstream);

    unordered_particles.resize(newnp);

    redistribute.recv_unpack(unordered_particles.data, newnp, mainstream, host_idle_time);

    localcomm.barrier();	// peh: +2

    particles.resize(newnp);

    cells.build(particles.xyzuvw.data, particles.size, mainstream, NULL, unordered_particles.data);

    int nrbcs;
    if (rbcscoll)
	nrbcs = redistribute_rbcs.post();

    int nctcs;
    if (ctcscoll)
	nctcs = redistribute_ctcs.post();

    if (rbcscoll)
	rbcscoll->resize(nrbcs);

    if (ctcscoll)
	ctcscoll->resize(nctcs);

    CUDA_CHECK(cudaPeekAtLastError());


    if (rbcscoll)
	redistribute_rbcs.unpack(rbcscoll->data(), rbcscoll->count(), mainstream);

    if (ctcscoll)
	redistribute_ctcs.unpack(ctcscoll->data(), ctcscoll->count(), mainstream);

    CUDA_CHECK(cudaPeekAtLastError());

//  localcomm.barrier();  // peh: +3

    timings["lockstep"] += MPI_Wtime() - tstart;
}


void Simulation::run()
{
    if (rank == 0 && !walls)
	printf("the simulation begins now and it consists of %.3e steps\n", (double)nsteps);	  
    
    double time_simulation_start = MPI_Wtime();
    
    _redistribute();
    _forces();
    
    if (!walls && pushtheflow)
	driving_acceleration = hydrostatic_a;		    
    
    particles.update_stage1(driving_acceleration, mainstream);
    
    if (rbcscoll)
	rbcscoll->update_stage1(driving_acceleration, mainstream);
    
    if (ctcscoll)
	ctcscoll->update_stage1(driving_acceleration, mainstream);
    
    int it;
    enum { nvtxstart = 8001, nvtxstop = 8051 } ;
    
    for(it = 0; it < nsteps; ++it)
    {
	const bool verbose = it > 0 && rank == 0;
	
#ifdef _USE_NVTX_
	if (it == nvtxstart)
	{
	    NvtxTracer::currently_profiling = true;
	    CUDA_CHECK(cudaProfilerStart());
	}
	else if (it == nvtxstop)
	{
	    CUDA_CHECK(cudaProfilerStop());
	    NvtxTracer::currently_profiling = false;
	    CUDA_CHECK(cudaDeviceSynchronize());
	    
	    if (rank == 0)
		printf("profiling session ended. terminating the simulation now...\n");

	    break;
	}
#endif
	
	if (it % steps_per_report == 0)
	{
	    CUDA_CHECK(cudaStreamSynchronize(mainstream));
	    
	    if (check_termination())
		break;
	    
	    _report(verbose, it);
	}
	
	_redistribute();

#if 1
    lockstep_check:

	const bool lockstep_OK =
	    !(walls && it >= wall_creation_stepid && wall == NULL) &&
	    !(it % steps_per_dump == 0) &&
	    !(it + 1 == nvtxstart) &&
	    !(it + 1 == nvtxstop) &&
	    !((it + 1) % steps_per_report == 0) &&
	    !(it + 1 == nsteps);

	if (lockstep_OK)
	{
	    _lockstep();

	    ++it;

	    goto lockstep_check;
	}
#endif
	
	if (walls && it >= wall_creation_stepid && wall == NULL)
	{
	    CUDA_CHECK(cudaDeviceSynchronize());
	    
	    bool termination_request = false;

	    _create_walls(verbose, termination_request);

	    _redistribute();
	    
	    if (termination_request)
		break;
	    
	    time_simulation_start = MPI_Wtime();
	    
	    if (pushtheflow)
		driving_acceleration = hydrostatic_a;
	    
	    if (rank == 0)
		printf("the simulation begins now and it consists of %.3e steps\n", (double)(nsteps - it));
	}
	
	_forces();
	
	if (it % steps_per_dump == 0)
	{
	    CUDA_CHECK(cudaStreamSynchronize(mainstream));
	    
	    _data_dump(it);
	}
	
	_update_and_bounce();
    }
    
    const double time_simulation_stop = MPI_Wtime();
    const double telapsed = time_simulation_stop - time_simulation_start;
    
    if (rank == 0)
	if (it == nsteps)
	    printf("simulation is done after %.2lf s (%dm%ds). Ciao.\n",
		   telapsed, (int)(telapsed / 60), (int)(telapsed) % 60);
	else
	    if (it != wall_creation_stepid)
		printf("external termination request (signal) after %.3e s. Bye.\n", telapsed);
    
    fflush(stdout);
}

Simulation::~Simulation()
{
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
