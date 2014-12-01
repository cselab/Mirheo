#include <string>
#include <sstream>
#include <vector>

#ifndef NO_H5PART
#include <H5Part.h>
#endif

#include "common.h"

using namespace std;

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;


H5PartDump::H5PartDump(const string fname, MPI_Comm cartcomm, const int L):
    cartcomm(cartcomm), fname(fname), tstamp(0)
{
#ifndef NO_H5PART
    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int c = 0; c < 3; ++c)
	origin[c] = L / 2 + coords[c] * L;

    H5PartFile * f = H5PartOpenFileParallel(fname.c_str(), H5PART_WRITE, cartcomm);

    assert(f != NULL);

handler = f;
#endif
}

void H5PartDump::dump(Particle * host_particles, int n)
{
#ifndef NO_H5PART
    H5PartFile * f = (H5PartFile *)handler;

    H5PartSetStep(f, tstamp);

    H5PartSetNumParticles(f, n);

    string labels[] = {"x", "y", "z"};

    for(int c = 0; c < 3; ++c)
    {
	vector<float> data(n);

	for(int i = 0; i < n; ++i)
	    data[i] = host_particles[i].x[c] + origin[c];

	H5PartWriteDataFloat32(f, labels[c].c_str(), &data.front());
    }

tstamp++;
#endif
}
    
H5PartDump::~H5PartDump()
{
#ifndef NO_H5PART
    H5PartFile * f = (H5PartFile *)handler;

H5PartCloseFile(f);
#endif
}

void xyz_dump(MPI_Comm comm, const char * filename, const char * particlename, Particle * particles, int n, int L, bool append)
{
    int rank;
    MPI_CHECK( MPI_Comm_rank(comm, &rank) );
    
    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(comm, 3, dims, periods, coords) );

    const int nlocal = n;
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );
    
    MPI_File f;
    MPI_CHECK( MPI_File_open(comm, filename , MPI_MODE_WRONLY | (append ? MPI_MODE_APPEND : MPI_MODE_CREATE), MPI_INFO_NULL, &f) );

    if (!append)
	MPI_CHECK( MPI_File_set_size (f, 0));
	
    MPI_Offset base;
    MPI_CHECK( MPI_File_get_position(f, &base));
	
    std::stringstream ss;

    if (rank == 0)
    {
	ss <<  n << "\n";
	ss << particlename << "\n";

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
	
    MPI_Status status;
	
    MPI_CHECK( MPI_File_write_at_all(f, base + offset, const_cast<char *>(content.c_str()), len, MPI_CHAR, &status));
	
    MPI_CHECK( MPI_File_close(&f));
}

void diagnostics(MPI_Comm comm, Particle * particles, int n, float dt, int idstep, int L, Acceleration * acc, bool particledump)
{
    const int nlocal = n;
    
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
	bool filenotthere;
	if (rank == 0)
	    filenotthere = access( "trajectories.xyz", F_OK ) == -1;

	MPI_CHECK( MPI_Bcast(&filenotthere, 1, MPI_INT, 0, comm) );
	
	static bool firsttime = true;

	firsttime |= filenotthere;

	xyz_dump(comm, "trajectories.xyz", "dpd-particles", particles, nlocal, L, !firsttime);

	firsttime = false;	
    }
}