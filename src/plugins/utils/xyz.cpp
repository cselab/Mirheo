#include "xyz.h"

void writeXYZ(MPI_Comm comm, std::string fname, Particle* particles, int np)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    int n = np;
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &n, &n, 1, MPI_INT, MPI_SUM, 0, comm) );

    MPI_File f;
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_CREATE|MPI_MODE_DELETE_ON_CLOSE|MPI_MODE_WRONLY, MPI_INFO_NULL, &f) );
    MPI_Check( MPI_File_close(&f) );
    MPI_Check( MPI_File_open(comm, fname.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f) );

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(5);

    if (rank == 0)
    {
        ss <<  n << "\n";
        ss << "# created by uDeviceX" << "\n";

        info("xyz dump to %s: total number of particles: %d", fname.c_str(), n);
    }

    for(int i = 0; i < np; ++i)
    {
        Particle& p = particles[i];

        ss << rank << " "
                << std::setw(10) << p.r.x << " "
                << std::setw(10) << p.r.y << " "
                << std::setw(10) << p.r.z << "\n";
    }

    std::string content = ss.str();

    MPI_Offset len = content.size();
    MPI_Offset offset = 0;
    MPI_Check( MPI_Exscan(&len, &offset, 1, MPI_OFFSET, MPI_SUM, comm));

    MPI_Status status;
    MPI_Check( MPI_File_write_at_all(f, offset, content.c_str(), len, MPI_CHAR, &status) );
    MPI_Check( MPI_File_close(&f));
}
