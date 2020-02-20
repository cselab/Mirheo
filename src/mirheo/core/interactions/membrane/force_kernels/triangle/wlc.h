#pragma once

#include "../parameters.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/reflection.h>
#include <mirheo/core/mesh/membrane.h>

#include <cmath>

namespace mirheo
{

/** \brief Compute shear energy on a given triangle with the Lim model.
    \tparam stressFreeState States if there is a stress free mesh associated with the interaction
 */
template <StressFreeState stressFreeState>
class TriangleWLCForce
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip breathe anoying warnings

    struct LengthArea
    {
        mReal l; ///< eq. edge length
        mReal a; ///< eq. triangle area
    };

    using EquilibriumTriangleDesc = LengthArea;
    using ParametersType          = WLCParameters;
#endif // DOXYGEN_SHOULD_SKIP_THIS
    
    /** \brief Construct the functor 
        \param [in] p The parameters of the model
        \param [in] mesh Triangle mesh information
        \param [in] lscale Scaling length factor, applied to all parameters
    */
    TriangleWLCForce(ParametersType p, const Mesh *mesh, mReal lscale) :
        lscale_(lscale)
    {
        x0_   = p.x0;
        ks_   = p.ks * lscale_ * lscale_;
        mpow_ = p.mpow;

        kd_ = p.kd * lscale_ * lscale_;
        
        area0_   = p.totArea0 * lscale_ * lscale_ / mesh->getNtriangles();
        length0_ = math::sqrt(area0_ * 4.0 / math::sqrt(3.0));
    }

    /** \brief Get the reference triangle information
        \param [in] mesh Mesh view that contains the reference mesh. Only used when stressFreeState is Active.
        \param [in] i0 Index (in the adjacent vertex ids space, see \c Mesh) of the first adjacent vertex
        \param [in] i1 Index (in the adjacent vertex ids space, see \c Mesh) of the second adjacent vertex
        \return The reference triangle information.
     */
    __D__ inline EquilibriumTriangleDesc getEquilibriumDesc(const MembraneMeshView& mesh, int i0, __UNUSED int i1) const
    {
        LengthArea eq;
        if (stressFreeState == StressFreeState::Active)
        {
            eq.l = mesh.initialLengths[i0] * lscale_;
            eq.a = mesh.initialAreas  [i0] * lscale_;
        }
        else
        {
            eq.l = length0_;
            eq.a = area0_;
        }
        return eq;
    }

    /** \brief Compute the triangle force on \p v1. See Developer docs for more details.
        \param [in] v1 vertex 1
        \param [in] v2 vertex 2
        \param [in] v3 vertex 3
        \param [in] eq The reference triangle information
        \return The triangle force acting on \p v1
     */
    __D__ inline mReal3 operator()(mReal3 v1, mReal3 v2, mReal3 v3, EquilibriumTriangleDesc eq) const
    {
        return _areaForce(v1, v2, v3, eq.a) + _bondForce(v1, v2, eq.l);
    }
        
private:

    __D__ mReal3 _bondForce(mReal3 v1, mReal3 v2, mReal l0) const
    {
        const mReal r = math::max(length(v2 - v1), 1e-5_mr);
        const mReal lmax     = l0 / x0_;
        const mReal inv_lmax = x0_ / l0;

        auto wlc = [this, inv_lmax] (mReal x) {
            return ks_ * inv_lmax * (4.0_mr*x*x - 9.0_mr*x + 6.0_mr) / ( 4.0_mr * sqr(1.0_mr - x) );
        };

        const mReal IbforceI_wlc = wlc( math::min(lmax - 1e-6_mr, r) * inv_lmax );

        const mReal kp = wlc( l0 * inv_lmax ) * fastPower(l0, mpow_+1);

        const mReal IbforceI_pow = -kp / (fastPower(r, mpow_+1));

        const mReal IfI = math::min(forceCap_, math::max(-forceCap_, IbforceI_wlc + IbforceI_pow));

        return IfI * (v2 - v1);
    }

    __D__ mReal3 _areaForce(mReal3 v1, mReal3 v2, mReal3 v3, mReal area0) const
    {
        const mReal3 x21 = v2 - v1;
        const mReal3 x32 = v3 - v2;
        const mReal3 x31 = v3 - v1;

        const mReal3 normalArea2 = cross(x21, x31);

        const mReal area = 0.5_mr * length(normalArea2);

        const mReal coef = kd_ * (area - area0) / (area * area0);

        return -0.25_mr * coef * cross(normalArea2, x32);
    }


    static constexpr mReal forceCap_ = 1500.0_mr;

    mReal x0_;
    mReal ks_;
    mReal mpow_;
    mReal kd_;

    mReal length0_, area0_; ///< only useful when StressFree is false
    mReal lscale_;
};

/// set type name
MIRHEO_TYPE_NAME(TriangleWLCForce<StressFreeState::Active>, "TriangleWCLForce<Active>");
/// set type name
MIRHEO_TYPE_NAME(TriangleWLCForce<StressFreeState::Inactive>, "TriangleWCLForce<Inactive>");

} // namespace mirheo
