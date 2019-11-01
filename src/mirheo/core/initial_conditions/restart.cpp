#include "restart.h"
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

RestartIC::RestartIC(std::string path) :
    path(path)
{}

RestartIC::~RestartIC() = default;

void RestartIC::exec(const MPI_Comm& comm, ParticleVector *pv, __UNUSED cudaStream_t stream)
{
    pv->restart(comm, path);
}



} // namespace mirheo
