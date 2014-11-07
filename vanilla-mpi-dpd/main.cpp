#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cassert>

#include <mpi.h>

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

#include "common.h"
#include "dpd-interactions.h"
#include "redistribute-particles.h"

using namespace std;

//velocity verlet stages
void update_stage1(Particle * p, Acceleration * a, int n, float dt)
{
    for(int i = 0; i < n; ++i)
    {
	for(int c = 0; c < 3; ++c)
	    p[i].u[c] += a[i].a[c] * dt * 0.5;

	for(int c = 0; c < 3; ++c)
	    p[i].x[c] += p[i].u[c] * dt;
    }
}

void update_stage2(Particle * p, Acceleration * a, int n, float dt)
{
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	    p[i].u[c] += a[i].a[c] * dt * 0.5;
}

void diagnostics(MPI_Comm comm, Particle * particles, int n, float dt, int idstep)
{
    int nlocal = n;
    
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
    {
	p[0] += particles[i].u[0];
	p[1] += particles[i].u[1];
	p[2] += particles[i].u[2];
    }

    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );
    
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, &p, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    
    double ke = 0;
    for(int i = 0; i < n; ++i)
	ke += pow(particles[i].u[0], 2) + pow(particles[i].u[1], 2) + pow(particles[i].u[2], 2);

    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &ke, &ke, 1, MPI_DOUBLE, MPI_SUM, 0, comm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );
    
    double kbt = 0.5 * ke / (n * 3. / 2);

    //output temperature and total momentum
    if (rank == 0)
    {
	FILE * f = fopen("diag.txt", idstep ? "a" : "w");

	if (idstep == 0)
	    fprintf(f, "TSTEP\tKBT\tPX\tPY\tPZ\n");
	
	fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
	
	fclose(f);
    }
    
    //output VMD file
    {
	std::stringstream ss;

	if (rank == 0)
	{
	    ss << n << "\n";
	    ss << "dpdparticle\n";
	}
    
	for(int i = 0; i < nlocal; ++i)
	    ss << "1 " << particles[i].x[0] << " " << particles[i].x[1] << " " << particles[i].x[2] << "\n";

	string content = ss.str();

	int len = content.size();
	int offset;
	MPI_CHECK( MPI_Exscan(&len, &offset, 1, MPI_INTEGER, MPI_SUM, comm)); 
	
	MPI_File f;
	char fn[] = "trajectories.xyz";
	MPI_CHECK( MPI_File_open(comm, fn, MPI_MODE_WRONLY | (idstep == 0 ? MPI_MODE_CREATE : MPI_MODE_APPEND), MPI_INFO_NULL, &f) );

	if (idstep == 0)
	    MPI_CHECK( MPI_File_set_size (f, 0));

	MPI_Offset base;
	MPI_CHECK( MPI_File_get_position(f, &base));
	
	MPI_Status status;
	MPI_CHECK( MPI_File_write_at_all(f, base + offset, const_cast<char *>(content.data()), len, MPI_CHAR, &status));
	
	MPI_CHECK( MPI_File_close(&f));
    }
}

int main(int argc, char ** argv)
{
    int ranks[3];
    
    if (argc != 4)
    {
	printf("usage: ./mpi-dpd <xranks> <yranks> <zranks>\n");
	exit(-1);
    }
    else
    {
	for(int i = 0; i < 3; ++i)
	    ranks[i] = atoi(argv[1 + i]);
    }
    
    MPI_CHECK( MPI_Init(&argc, &argv) );

    int nranks, rank;
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );

    MPI_Comm cartcomm;
    
    int periods[] = {1,1,1};
    MPI_CHECK( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 1, &cartcomm) );

    const int L = 8;

    vector<Particle> particles(L * L * L * 3);

    for(auto& p : particles)
	for(int c = 0; c < 3; ++c)
	    p.x[c] = -L * 0.5 + drand48() * L;

    RedistributeParticles redistribute(cartcomm, L);
    
    const size_t nsteps = (int)(tend / dt);

    int domainsize[3];
    for(int i = 0; i < 3; ++i)
	domainsize[i] = L * ranks[i];

    vector<Acceleration> accel(particles.size());

    int saru_tag = rank;

    ComputeInteractionsDPD dpd(cartcomm, L);
    dpd.evaluate(saru_tag, &particles.front(), particles.size(), &accel.front());
    
    for(int it = 0; it < nsteps; ++it)
    {
	if (rank == 0)
	    printf("beginning of time step %d\n", it);
	
	update_stage1(&particles.front(), &accel.front(), particles.size(), dt);
	
	int newnp = redistribute.stage1(&particles.front(), particles.size());

	particles.resize(newnp);
	accel.resize(newnp);

	redistribute.stage2(&particles.front(), particles.size());

	dpd.evaluate(saru_tag, &particles.front(), particles.size(), &accel.front());

	update_stage2(&particles.front(), &accel.front(), particles.size(), dt);

	diagnostics(cartcomm, &particles.front(), particles.size(), dt, it);
    }
    
    MPI_CHECK( MPI_Finalize() );

    if (rank == 0)
	printf("simulation is done\n");
    
    return 0;
}
	
