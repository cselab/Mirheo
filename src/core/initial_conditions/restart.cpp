#include "restart.h"
#include <core/pvs/particle_vector.h>

RestartIC::RestartIC(std::string path) :
    path(path)
{}

RestartIC::~RestartIC() = default;

void RestartIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    pv->restart(comm, path);
}


