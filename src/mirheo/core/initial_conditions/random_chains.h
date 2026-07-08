// Copyright 2022 ETH Zurich. All Rights Reserved.
#pragma once

#include "interface.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/domain.h>

#include <mpi.h>
#include <vector>

namespace mirheo
{

class ParticleVector;

/** \brief Initialize ChainVector objects

    Initialize chain objects from center of mass positions.
    Each chain is generated from a random walk in 3D.
*/
class RandomChainsIC : public InitialConditions
{
public:
    /** \brief Construct a RandomChainsIC object
        \param [in] positions List of center of mass of each chain.
        The size of the list is the number of chains that will be initialized.
        \param [in] length Length of a single chain.
    */
    RandomChainsIC(std::vector<real3> positions, real length);
    ~RandomChainsIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<real3> positions_;
    real length_;
};

} // namespace mirheo
