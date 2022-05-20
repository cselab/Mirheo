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

    Initialize chain objects from center of mass positions and unit vector orientations.
*/
class ChainIC : public InitialConditions
{
public:
    /** \brief Construct a ChainIC object
        \param [in] positions List of center of mass of each chain.
        The size of the list is the number of chains that will be initialized.
        \param [in] orientations List of orientations of each object. Must have the same length as `positions`.
        \param [in] length Length of a single chain.
    */
    ChainIC(std::vector<real3> positions, std::vector<real3> orientations, real length);
    ~ChainIC();

    void exec(const MPI_Comm& comm, ParticleVector *pv, cudaStream_t stream) override;

private:
    std::vector<real3> positions_;
    std::vector<real3> orientations_;
    real length_;
};

} // namespace mirheo
