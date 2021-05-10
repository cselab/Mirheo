// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "../fetchers.h"
#include "../parameters.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

#include <cmath>

namespace mirheo
{
/// Compute bending forces from the Kantor model.
class DihedralKantor : public VertexFetcher
{
public:
    /// Type of parameters that describe the kernel
    using ParametersType = KantorBendingParameters;

    /** \brief Initialize the functor
        \param [in] p The parameters of the functor
        \param [in] lscale Scaling length factor, applied to all parameters
     */
    DihedralKantor(ParametersType p, mReal lscale)
    {
        const mReal theta0 = p.theta / 180.0 * M_PI;

        cost0kb_ = math::cos(theta0) * p.kb;
        sint0kb_ = math::sin(theta0) * p.kb;

        applyLengthScalingFactor(lscale);
    }

    /// Scale length-dependent parameters.
    __HD__ void applyLengthScalingFactor(mReal lscale)
    {
        cost0kb_ *= lscale * lscale;
        sint0kb_ *= lscale * lscale;
    }

    /// Precompute internal values that are common to all vertices in the cell.
    __D__ inline void computeInternalCommonQuantities(const ViewType& view, int rbcId)
    {}

    /** \brief Compute the dihedral forces. See Developer docs for more details.
        \param [in] v0 vertex 0
        \param [in] v1 vertex 1
        \param [in] v2 vertex 2
        \param [in] v3 vertex 3
        \param [in,out] f1 force acting on \p v1; this method will add (not set) the dihedral force to that quantity.
        \return The dihedral force acting on \p v0
     */
    __D__ inline mReal3 operator()(VertexType v0, VertexType v1, VertexType v2, VertexType v3, mReal3 &f1) const
    {
        return _kantor(v1, v0, v2, v3, f1);
    }

private:

    __D__ inline mReal3 _kantor(VertexType v1, VertexType v2, VertexType v3, VertexType v4, mReal3 &f1) const
    {
        const mReal3 ksi   = cross(v1 - v2, v1 - v3);
        const mReal3 dzeta = cross(v3 - v4, v2 - v4);

        const mReal overIksiI   = math::rsqrt(dot(ksi, ksi));
        const mReal overIdzetaI = math::rsqrt(dot(dzeta, dzeta));

        const mReal cosTheta = dot(ksi, dzeta) * overIksiI * overIdzetaI;
        const mReal IsinThetaI2 = 1.0_mr - cosTheta*cosTheta;

        const mReal rawST_1 = math::rsqrt(max(IsinThetaI2, 1.0e-6_mr));
        const mReal sinTheta_1 = copysignf( rawST_1, dot(ksi - dzeta, v4 - v1) ); // because the normals look inside
        const mReal beta = cost0kb_ - cosTheta * sint0kb_ * sinTheta_1;

        const mReal b11 = -beta * cosTheta *  overIksiI   * overIksiI;
        const mReal b12 =  beta *             overIksiI   * overIdzetaI;
        const mReal b22 = -beta * cosTheta *  overIdzetaI * overIdzetaI;

        f1 = cross(ksi, v3 - v2)*b11 + cross(dzeta, v3 - v2)*b12;

        return cross(ksi, v1 - v3)*b11 + ( cross(ksi, v3 - v4) + cross(dzeta, v1 - v3) )*b12 + cross(dzeta, v3 - v4)*b22;
    }

    mReal cost0kb_; ///< kb * cos(theta_0)
    mReal sint0kb_; ///< kb * sin(theta_0)
};

/// create name for that type
MIRHEO_TYPE_NAME_AUTO(DihedralKantor);

} // namespace mirheo
