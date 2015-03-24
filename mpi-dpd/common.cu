/*
 *  common.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-01-30.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <sys/resource.h>

#include <cuda-dpd.h>

#include "common.h"

#ifdef _USE_NVTX_
bool NvtxTracer::currently_profiling = false;
#endif

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;

bool Acceleration::initialized = false;

MPI_Datatype Acceleration::mytype;

void CellLists::build(Particle * const p, const int n, cudaStream_t stream)
{
    if (n > 0)
	build_clists((float * )p, n, 1, LX, LY, LZ, -LX/2, -LY/2, -LZ/2, NULL, start, count,  NULL, 0);
    else
    {
	CUDA_CHECK(cudaMemset(start, 0, sizeof(int) * ncells));
	CUDA_CHECK(cudaMemset(count, 0, sizeof(int) * ncells));
    }
}

void report_host_memory_usage(MPI_Comm comm, FILE * foutput)
{
    struct rusage rusage;
    long peak_rss;

    getrusage(RUSAGE_SELF, &rusage);
    peak_rss = rusage.ru_maxrss*1024;
    
    long rss = 0;
    FILE* fp = NULL;
    if ( (fp = fopen( "/proc/self/statm", "r" )) == NULL ) 
    {
	return;
    }

    if ( fscanf( fp, "%*s%ld", &rss ) != 1 )
    {
	fclose( fp );
	return;
    }
    fclose( fp );

    long current_rss;

    current_rss = rss * sysconf( _SC_PAGESIZE);
    
    long max_peak_rss, sum_peak_rss;
    long max_current_rss, sum_current_rss;

    MPI_Reduce(&peak_rss, &max_peak_rss, 1, MPI_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&peak_rss, &sum_peak_rss, 1, MPI_LONG, MPI_SUM, 0, comm);
    MPI_Reduce(&current_rss, &max_current_rss, 1, MPI_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&current_rss, &sum_current_rss, 1, MPI_LONG, MPI_SUM, 0, comm);

    int rank;
    MPI_Comm_rank(comm, &rank);

    if (rank == 0)
    {
	fprintf(foutput, "> peak resident set size: max = %.2lf Mbytes sum = %.2lf Mbytes\n",
		max_peak_rss/(1024.0*1024.0), sum_peak_rss/(1024.0*1024.0));
	fprintf(foutput, "> current resident set size: max = %.2lf Mbytes sum = %.2lf Mbytes\n",
		max_current_rss/(1024.0*1024.0), sum_current_rss/(1024.0*1024.0));
    }
}

void diagnostics(MPI_Comm comm, MPI_Comm cartcomm, Particle * particles, int n, float dt, int idstep, Acceleration * acc)
{
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	    p[c] += particles[i].u[c];

    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );

    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );
    
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, rank == 0 ? &p : NULL, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    
    double ke = 0;
    for(int i = 0; i < n; ++i)
	ke += pow(particles[i].u[0], 2) + pow(particles[i].u[1], 2) + pow(particles[i].u[2], 2);

    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );
    
    double kbt = 0.5 * ke / (n * 3. / 2);

    if (rank == 0)
    {
	static bool firsttime = true;
	FILE * f = fopen("diag.txt", firsttime ? "w" : "a");
	firsttime = false;
	
	if (idstep == 0)
	    fprintf(f, "TSTEP\tKBT\tPX\tPY\tPZ\n");

	printf("\x1b[91m%e\t%.10e\t%.10e\t%.10e\t%.10e\x1b[0m\n", idstep * dt, kbt, p[0], p[1], p[2]);
	fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
		
	fclose(f);
    }
}