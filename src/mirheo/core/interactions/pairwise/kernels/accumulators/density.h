// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{
/// \brief Accumulate densities on device
class DensityAccumulator
{
public:
    /// \brief Initialize the DensityAccumulator
    __D__ DensityAccumulator() :
        den_(0._r)
    {}

    /** \brief Atomically add density \p d to the destination \p view at id \p id.
        \param [in] d The value to add
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    __D__ void atomicAddToDst(real d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    /** \brief Atomically add density \p d to the source \p view at id \p id.
        \param [in] d The value to add
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    __D__ void atomicAddToSrc(real d, PVviewWithDensities& view, int id) const
    {
        atomicAdd(view.densities + id, d);
    }

    /// \return the internal accumulated density
    __D__ real get() const {return den_;}

    /// add \p d to the internal density
    __D__ void add(real d) {den_ += d;}

private:
    real den_; ///< internal accumulated density
};

} // namespace mirheo
