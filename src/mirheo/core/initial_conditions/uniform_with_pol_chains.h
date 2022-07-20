// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"
#include <mirheo/core/datatypes.h>

namespace mirheo
{

/** \brief Fill the domain with uniform number density and polymer chains of zero length

    Initialize particles uniformly with the given number density on the whole domain.
    The domain considered is that of the ParticleVector.
    ObjectVector objects are not supported.
 */
class UniformWithPolChainIC : public InitialConditions
{
public:

    /** \brief Construct a UniformWithPolChainIC object
        \param [in] numDensity Number density of the particles to initialize
     */
    UniformWithPolChainIC(real numDensity);

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    real numDensity_;
};


} // namespace mirheo
