#include "restart.h"
#include <mirheo/core/pvs/particle_vector.h>

namespace mirheo
{

RestartIC::RestartIC(const std::string& path) :
    path_(path)
{}

RestartIC::~RestartIC() = default;

void RestartIC::exec(const MPI_Comm& comm, ParticleVector *pv, __UNUSED cudaStream_t stream)
{
    pv->restart(comm, path_);
}



} // namespace mirheo
