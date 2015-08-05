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

bool is_mps_enabled;

#ifdef _USE_NVTX_
bool NvtxTracer::currently_profiling = false;
#endif

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;

bool Acceleration::initialized = false;

MPI_Datatype Acceleration::mytype;

void CellLists::build(Particle * const p, const int n, cudaStream_t stream, int * const order, const Particle * const src)
{
    NVTX_RANGE("Cells-build", NVTX_C1)

	if (n > 0)
	{
	    if (!is_mps_enabled || XSIZE_SUBDOMAIN < 64 && YSIZE_SUBDOMAIN < 64 && ZSIZE_SUBDOMAIN < 64)
		build_clists((float * )p, n, 1, LX, LY, LZ, -LX/2, -LY/2, -LZ/2, order, start, count,  NULL, stream, (float *)src);
	    else
		build_clists_vanilla((float * )p, n, 1, LX, LY, LZ, -LX/2, -LY/2, -LZ/2, order, start, count,  NULL, stream, (float *)src);
	}
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

inline int Adler32(const void *buf, size_t len)
{
    const char * data = (const char *)buf;
    int a = 1, b = 0;
    const int MOD_ADLER = 65521;

    for (size_t index = 0; index < len; ++index)
    {
	a = (a + data[index]) % MOD_ADLER;
	b = (b + a) % MOD_ADLER;
    }

    return (b << 16) | a;
}

LocalComm::LocalComm()
{
    local_comm = MPI_COMM_NULL;
    local_rank = 0;
    local_nranks = 1;
}

void LocalComm::initialize(MPI_Comm active_comm)
{
    MPI_Comm_rank(active_comm, &rank);
    MPI_Comm_size(active_comm, &nranks);

    local_comm = active_comm;

    MPI_Get_processor_name(name, &len);
    int id = Adler32(name, len);

    MPI_Comm_split(active_comm, id, rank, &local_comm) ;

    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_nranks);
}

void LocalComm::barrier()
{
    if (!is_mps_enabled || local_nranks == 1) return;

    MPI_CHECK(MPI_Barrier(local_comm));
}

void LocalComm::print_particles(int np)
{
    //if (!cuda_mps_enabled || local_nranks == 1) return;

    int node_np;
    MPI_Reduce(&np, &node_np, 1, MPI_INT, MPI_SUM, 0, local_comm);
    if (local_rank == 0) {
	printf("Grank: %d node: %s particles: %d\n", rank, name, node_np); fflush(0);
    }
}
