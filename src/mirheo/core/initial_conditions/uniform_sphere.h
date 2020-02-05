#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <vector_types.h>

namespace mirheo
{

/** \brief Fill the domain with uniform number density in a given ball
    \ingroup ICs
    
    Initialize particles uniformly with the given number density inside or outside a ball.
    The domain considered is that of the ParticleVector.
    ObjectVector objects are not supported.
 */
class UniformSphereIC : public InitialConditions
{
public:
    /** \brief Construct a UniformSphereIC object
        \param [in] numDensity Number density of the particles to initialize
        \param [in] center Center of the ball
        \param [in] radius Radius of the ball
        \param [in] inside The particles will be inside the ball if set to `true`, outside otherwise.
     */
    UniformSphereIC(real numDensity, real3 center, real radius, bool inside);
    ~UniformSphereIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    real numDensity_;
    real3 center_;
    real  radius_;
    bool inside_;
};

} // namespace mirheo
