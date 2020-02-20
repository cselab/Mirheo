#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <vector_types.h>
#include <vector>

namespace mirheo
{
/** \brief Initialize particles to given positions and velocities.

    ObjectVector objects are not supported.
*/
class FromArrayIC : public InitialConditions
{
public:
    /** \brief Construct a FromArrayIC object
        \param [in] pos list of initial positions in global coordinates. 
                    The size will determine the maximum number of particles.
                    Positions outside the domain are filtered out.
        \param [in] vel list of initial velocities.
                    Must have the same size as \p pos.
     */
    FromArrayIC(const std::vector<real3>& pos, const std::vector<real3>& vel);

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<real3> pos_, vel_;
};


} // namespace mirheo
