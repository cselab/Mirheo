// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/pvs/views/pv_with_stresses.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/// Holds force and stress together
struct ForceStress
{
    real3 force; ///< force value
    Stress stress; ///< stress value
};

/** \brief Accumulate ForceStress structure on device
    \tparam BasicView The view type without stress, to enforce the use of the stress view wrapper
 */
template <typename BasicView>
class ForceStressAccumulator
{
public:
    /// \brief Initialize the ForceStressAccumulator
    __D__ ForceStressAccumulator() :
        frcStress_({{0._r, 0._r, 0._r},
                    {0._r, 0._r, 0._r, 0._r, 0._r, 0._r}})
    {}

    /** \brief Atomically add the force and stress \p fs to the destination \p view at id \p id.
        \param [in] fs The force, directed from src to dst, and the corresponding stress
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    __D__ void atomicAddToDst(const ForceStress& fs, PVviewWithStresses<BasicView>& view, int id) const
    {
        atomicAdd(      view.forces   + id, fs.force );
        atomicAddStress(view.stresses + id, fs.stress);
    }

    /** \brief Atomically add the force and stress \p fs to the source \p view at id \p id.
        \param [in] fs The force, directed from src to dst, and the corresponding stress
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    __D__ void atomicAddToSrc(const ForceStress& fs, PVviewWithStresses<BasicView>& view, int id) const
    {
        atomicAdd(      view.forces   + id, -fs.force );
        atomicAddStress(view.stresses + id,  fs.stress);
    }

    /// \return the internal accumulated force and stress
    __D__ ForceStress get() const {return frcStress_;}

    /// add \p fs to the internal force
    __D__ void add(const ForceStress& fs)
    {
        frcStress_.force += fs.force;
        frcStress_.stress.xx += fs.stress.xx;
        frcStress_.stress.xy += fs.stress.xy;
        frcStress_.stress.xz += fs.stress.xz;
        frcStress_.stress.yy += fs.stress.yy;
        frcStress_.stress.yz += fs.stress.yz;
        frcStress_.stress.zz += fs.stress.zz;
    }

private:
    ForceStress frcStress_; ///< internal accumulated force and stress

    /// addition wrapper for stresses; uses \c atomicAdd().
    __D__ void atomicAddStress(Stress *dst, const Stress& s) const
    {
        atomicAdd(&dst->xx, s.xx);
        atomicAdd(&dst->xy, s.xy);
        atomicAdd(&dst->xz, s.xz);
        atomicAdd(&dst->yy, s.yy);
        atomicAdd(&dst->yz, s.yz);
        atomicAdd(&dst->zz, s.zz);
    }
};

} // namespace mirheo
