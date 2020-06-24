// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{
/// \brief Accumulate forces on device
class ForceAccumulator
{
public:
    /// \brief Initialize the ForceAccumulator
    __D__ ForceAccumulator() :
        frc_({0._r, 0._r, 0._r})
    {}

    /** \brief Atomically add the force \p f to the destination \p view at id \p id.
        \param [in] f The force, directed from src to dst
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    __D__ void atomicAddToDst(real3 f, PVview& view, int id) const
    {
        atomicAdd(view.forces + id, f);
    }

    /** \brief Atomically add the force \p f to the source \p view at id \p id.
        \param [in] f The force, directed from src to dst
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    __D__ void atomicAddToSrc(real3 f, PVview& view, int id) const
    {
        atomicAdd(view.forces + id, -f);
    }

    /// \return the internal accumulated force
    __D__ real3 get() const {return frc_;}

    /// add \p f to the internal force
    __D__ void add(real3 f) {frc_ += f;}

private:
    real3 frc_;  ///< internal accumulated force
};

} // namespace mirheo
