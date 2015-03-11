#include <mpi.h>

#include <string>

#include "common.h"


void xyz_dump(MPI_Comm comm, MPI_Comm cartcomm, const char * filename, const char * particlename, Particle * particles, int n, bool append);

void ply_dump(MPI_Comm comm, MPI_Comm cartcomm, const char * filename,
	      int (*mesh_indices)[3], const int ninstances, const int ntriangles_per_instance, Particle * _particles, 
	      int nvertices_per_instance, bool append);

class H5PartDump
{
    float origin[3];

    void * handler;
    bool disposed;
    int tstamp;

    void _initialize(const std::string filename, MPI_Comm comm, MPI_Comm cartcomm);
    void _dispose();

public:

    H5PartDump(const std::string filename, MPI_Comm comm, MPI_Comm cartcomm);

    void close() { _dispose(); }

    void dump(Particle * host_particles, int n);
    
    ~H5PartDump();
};

class H5FieldDump
{
    int last_idtimestep, globalsize[3];

    MPI_Comm cartcomm;

    void _write_fields(const char * const path2h5,
		       const float * const channeldata[], const char * const * const channelnames, const int nchannels, 
		       MPI_Comm comm, const float time);
    
    void _xdmf_header(FILE * xmf);
    void _xdmf_grid(FILE * xmf, float time, const char * const h5path, const char * const * channelnames, int nchannels);
    void _xdmf_epilogue(FILE * xmf);

public:

    H5FieldDump(MPI_Comm cartcomm);

    void dump(MPI_Comm comm, const Particle * const p, const int n, int idtimestep);

    void dump_scalarfield(MPI_Comm comm, const float * const data, const char * channelname);

    ~H5FieldDump();
};
