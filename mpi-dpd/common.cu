#include <string>
#include <sstream>
#include <vector>

#include <H5Part.h>

#include "common.h"

using namespace std;

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;

void h5part_dump(MPI_Comm comm, Particle * host_particles, int n, int ntotal, float dt, int idstep, float xoffset, float yoffset, float zoffset)
{
    H5PartFile * f = H5PartOpenFileParallel("trajectories.h5", H5PART_WRITE, comm);

    assert(H5PartFileIsValid(f));

    H5PartSetStep(f, idstep);

    H5PartSetNumParticles(f, ntotal);

    string labels[] = {"x", "y", "z"};

    float offset[] = { xoffset, yoffset, zoffset };

    for(int c = 0; c < 3; ++c)
    {
	vector<float> data(n);

	for(int i = 0; i < n; ++i)
	    data[i] = host_particles[i].x[c] + offset[c];

	H5PartWriteDataFloat32(f, labels[c].c_str(), &data.front());
    }

    H5PartCloseFile(f);
}

void diagnostics(MPI_Comm comm, Particle * _particles, int n, float dt, int idstep, int L, Acceleration * _acc, bool particledump)
{
    Particle * particles = new Particle[n];
    Acceleration * acc = new Acceleration[n];

    CUDA_CHECK(cudaMemcpy(particles, _particles, sizeof(Particle) * n, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(acc, _acc, sizeof(Acceleration) * n, cudaMemcpyDeviceToHost));
    
    int nlocal = n;

    //we fused VV stages so we need to recover the state before stage 1
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	{
	    assert(!isnan(particles[i].x[c]));
	    assert(!isnan(particles[i].u[c]));
	    assert(!isnan(acc[i].a[c]));
	    
	    particles[i].x[c] -= dt * particles[i].u[c];
	    particles[i].u[c] -= 0.5 * dt * acc[i].a[c];
	}
    
    double p[] = {0, 0, 0};
    for(int i = 0; i < n; ++i)
	for(int c = 0; c < 3; ++c)
	    p[c] += particles[i].u[c];

    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );

    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(comm, 3, dims, periods, coords) );
    
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &p, rank == 0 ? &p : NULL, 3, MPI_DOUBLE, MPI_SUM, 0, comm) );
    
    if (rank == 0)
	printf("momentum: %f %f %f\n", p[0], p[1], p[2]);

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
	
	fprintf(f, "%e\t%.10e\t%.10e\t%.10e\t%.10e\n", idstep * dt, kbt, p[0], p[1], p[2]);
	
	fclose(f);
    }

    if (particledump)
    {
	h5part_dump(comm, particles, nlocal, n, dt, idstep, L / 2 + coords[0] * L, L / 2 + coords[1] * L, L / 2 + coords[2] * L);
	
	std::stringstream ss;

	if (rank == 0)
	{
	    ss <<  n << "\n";
	    ss << "dpdparticle\n";

	    printf("total number of particles: %d\n", n);
	}
    
	for(int i = 0; i < nlocal; ++i)
	    ss << rank << " " 
	       << (particles[i].x[0] + L / 2 + coords[0] * L) << " "
	       << (particles[i].x[1] + L / 2 + coords[1] * L) << " "
	       << (particles[i].x[2] + L / 2 + coords[2] * L) << "\n";

	string content = ss.str();
	
	int len = content.size();
	int offset = 0;
	MPI_CHECK( MPI_Exscan(&len, &offset, 1, MPI_INTEGER, MPI_SUM, comm)); 
	
	MPI_File f;
	char fn[] = "/scratch/daint/diegor/trajectories.xyz";
	
	static bool firsttime = true;

	MPI_CHECK( MPI_File_open(comm, fn, MPI_MODE_WRONLY | (firsttime ? MPI_MODE_CREATE : MPI_MODE_APPEND), MPI_INFO_NULL, &f) );

	if (firsttime)
	    MPI_CHECK( MPI_File_set_size (f, 0));

	firsttime = false;
	
	MPI_Offset base;
	MPI_CHECK( MPI_File_get_position(f, &base));
	
	MPI_Status status;
	
	MPI_CHECK( MPI_File_write_at_all(f, base + offset, const_cast<char *>(content.c_str()), len, MPI_CHAR, &status));
	
	MPI_CHECK( MPI_File_close(&f));
    }

    delete [] particles;
    delete [] acc;
}