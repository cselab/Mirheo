#pragma once

#include "helpers.h"
#include "interface.h"

#include <mirheo/core/datatypes.h>

#include <vector_types.h>

namespace mirheo
{

/** \brief Fill the domain with uniform number density in a given region
    
    Initialize particles uniformly with the given number density on a specified region of the domain.
    The region is specified by a filter functor.
    The domain considered is that of the ParticleVector.
    ObjectVector objects are not supported.
 */
class UniformFilteredIC : public InitialConditions
{
public:
    /** \brief Construct a UniformFilteredIC object
        \param [in] numDensity Number density of the particles to initialize
        \param [in] filter Indicator function that maps a position of the domain to a boolean value.
                           It returns \c true if the position is inside the region.
     */
    UniformFilteredIC(real numDensity, PositionFilter filter);
    ~UniformFilteredIC();
    
    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;    

private:
    real numDensity_;
    PositionFilter filter_;
};


} // namespace mirheo
