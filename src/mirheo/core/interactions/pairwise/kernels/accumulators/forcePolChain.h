// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/pvs/views/pv_with_pol_chain.h>
#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>

namespace mirheo
{

/// Holds force and time derivative of polymeric chain vector together
struct ForceDerPolChain
{
    real3 force; ///< force valu
    real3 dQdst_dt; ///< time derivative of Q at dst particle
    real3 dQsrc_dt; ///< time derivative of Q at src particle

    /** overloaded operator to add a generalized force.
        \param f the value to add
        \return The old value of the object
     */
    __D__ inline ForceDerPolChain& operator+=(const ForceDerPolChain& f)
    {
        this->force += f.force;
        this->dQdst_dt += f.dQdst_dt;
        this->dQsrc_dt += f.dQsrc_dt;
        return *this;
    }
};




/** \brief Accumulate ForceDerPolChain structure on device
 */
class ForceDerPolChainAccumulator
{
public:
    /// \brief Initialize the ForceDerPolChainAccumulator
    __D__ ForceDerPolChainAccumulator() :
        val_({{0._r, 0._r, 0._r},
              {0._r, 0._r, 0._r},
              {0._r, 0._r, 0._r}})
    {}

    /** \brief Atomically add the force and polymeric chain vector time derivative \p fq to the destination \p view at id \p id.
        \param [in] fq The force and dQdt
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    static __D__ void atomicAddToDst(const ForceDerPolChain& fq, PVviewWithPolChainVector& view, int id)
    {
        atomicAdd(view.forces+ id, fq.force );
        atomicAdd(view.dQdt  + id, fq.dQdst_dt);
    }

    /** \brief Atomically add the force and polymeric chain vector time derivative \p fq to the source \p view at id \p id.
        \param [in] fq The force and dQdt
        \param [out] view The destination container
        \param [in] id destination index in \p view
     */
    static __D__ void atomicAddToSrc(const ForceDerPolChain& fq, PVviewWithPolChainVector& view, int id)
    {
        atomicAdd(view.forces + id, -fq.force);
        atomicAdd(view.dQdt   + id,  fq.dQsrc_dt);
    }

    /// \return the internal accumulated force and stress
    __D__ ForceDerPolChain get() const {return val_;}

    /// add \p fq to the internal value (only to the dst particle)
    __D__ void add(const ForceDerPolChain& fq)
    {
        val_.force += fq.force;
        val_.dQdst_dt += fq.dQdst_dt;
    }

private:
    ForceDerPolChain val_; ///< internal accumulated force and polymeric chain vector derivatives
};

/** Get the force from a generalized force.
    \return force vector
 */
inline __D__ real3 getForce(ForceDerPolChain f) {return f.force;}

} // namespace mirheo
