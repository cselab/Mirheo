#include <mpi.h>

#include <string>

#include "common.h"


void xyz_dump(MPI_Comm comm, const char * filename, const char * particlename, Particle * particles, int n, bool append);

void ply_dump(MPI_Comm comm, const char * filename,
	      int (*mesh_indices)[3], const int ninstances, const int ntriangles_per_instance, Particle * _particles, 
	      int nvertices_per_instance, bool append);

class H5PartDump
{
    MPI_Comm cartcomm;

    std::string fname;

    float origin[3];

    void * handler;

    int tstamp;
public:

    H5PartDump(const std::string filename, MPI_Comm cartcomm);

    void dump(Particle * host_particles, int n);
    
    ~H5PartDump();
};

class H5FieldDump
{
    int last_idtimestep, globalsize[3];

    MPI_Comm cartcomm;

    void _write_fields(const char * const path2h5,
		       const float * const channeldata[], const char * const * const channelnames, const int nchannels, 
		       MPI_Comm cartcomm, const float time);
    
    void _xdmf_header(FILE * xmf);
    void _xdmf_grid(FILE * xmf, float time, const char * const h5path, const char * const * channelnames, int nchannels);
    void _xdmf_epilogue(FILE * xmf);

public:

    H5FieldDump(MPI_Comm cartcomm);

    void dump(const Particle * const p, const int n, int idtimestep);

    void dump_scalarfield(const float * const data, const char * channelname);

    ~H5FieldDump();
};
