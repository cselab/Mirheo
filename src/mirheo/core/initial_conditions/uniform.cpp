#include "uniform.h"
#include "helpers.h"

namespace mirheo
{

UniformIC::UniformIC(real numDensity) :
    numDensity_(numDensity)
{}

UniformIC::~UniformIC() = default;


/**
 * Particles will be initialized such that the number of particles \f$ n_p \f$ in
 * each unit cell of the domain follows:
 *
 * \f$
 * \begin{cases}
 *   p \left( n_p = \left\lfloor \rho \right\rfloor \right) = \left\lceil \rho \right\rceil - \rho \\
 *   p \left( n_p = \left\lceil \rho \right\rceil \right) = \rho - \left\lfloor \rho \right\rfloor
 * \end{cases}
 * \f$
 *
 * Here \f$ \rho \f$ is the target number density: #density
 *
 * Each particle will have a unique id across all MPI processes in Particle::i1.
 *
 * \rst
 * .. note::
 *    Currently ids are only 32-bit wide
 * \endrst
 */
void UniformIC::exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream)
{
    auto filterInKeepAll = [](real3) {return true;};
    addUniformParticles(numDensity_, comm, pv, filterInKeepAll, stream);
}

} // namespace mirheo
