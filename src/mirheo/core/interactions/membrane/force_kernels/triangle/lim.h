#pragma once

#include "../parameters.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/mesh/membrane.h>

#include <cmath>

namespace mirheo
{

/** \brief Compute shear energy on a given triangle with the Lim model.
    \tparam stressFreeState States if there is a stress free mesh associated with the interaction
 */
template <StressFreeState stressFreeState>
class TriangleLimForce
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // skip breathe anoying warnings
    /// Stores information of reference triangle
    struct LengthsArea
    {
        mReal l0;   ///< first  equilibrium edge length
        mReal l1;   ///< second equilibrium edge length
        mReal a;    ///< equilibrium triangle area
        mReal dotp; ///< dot product of the two above edges
    };
    
    using EquilibriumTriangleDesc = LengthsArea; ///< type to describe reference triangle info
    using ParametersType          = LimParameters; ///< type to describe parameters
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /** \brief Construct the functor 
        \param [in] p The parameters of the model
        \param [in] mesh Triangle mesh information
        \param [in] lscale Scaling length factor, applied to all parameters
    */
    TriangleLimForce(ParametersType p, const Mesh *mesh, mReal lscale) :
        lscale_(lscale)
    {
        a3_ = p.a3;
        a4_ = p.a4;
        b1_ = p.b1;
        b2_ = p.b2;

        ka_ = p.ka * lscale_ * lscale_;
        mu_ = p.mu * lscale_ * lscale_;
        
        area0_   = p.totArea0 * lscale_ * lscale_ / mesh->getNtriangles();
        length0_ = math::sqrt(area0_ * 4.0 / math::sqrt(3.0));
    }

    /** \brief Get the reference triangle information
        \param [in] mesh Mesh view that contains the reference mesh. Only used when stressFreeState is Active.
        \param [in] i0 Index (in the adjacent vertex ids space, see \c Mesh) of the first adjacent vertex
        \param [in] i1 Index (in the adjacent vertex ids space, see \c Mesh) of the second adjacent vertex
        \return The reference triangle information.
     */
    __D__ inline EquilibriumTriangleDesc getEquilibriumDesc(const MembraneMeshView& mesh, int i0, int i1) const
    {
        EquilibriumTriangleDesc eq;
        if (stressFreeState == StressFreeState::Active)
        {
            eq.l0   = mesh.initialLengths    [i0] * lscale_;
            eq.l1   = mesh.initialLengths    [i1] * lscale_;
            eq.a    = mesh.initialAreas      [i0] * lscale_ * lscale_;
            eq.dotp = mesh.initialDotProducts[i0] * lscale_ * lscale_;
        }
        else
        {
            eq.l0   = length0_;
            eq.l1   = length0_;
            eq.a    = area0_;
            eq.dotp = length0_ * 0.5_mr;
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
        const mReal3 x12 = v2 - v1;
        const mReal3 x13 = v3 - v1;
        const mReal3 x32 = v2 - v3;

        const mReal3 normalArea2 = cross(x12, x13);
        const mReal area = 0.5_mr * length(normalArea2);
        const mReal area_inv = 1.0_mr / area;
        const mReal area0_inv = 1.0_mr / eq.a;

        const mReal3 derArea  = (0.25_mr * area_inv) * cross(normalArea2, x32);

        const mReal alpha = area * area0_inv - 1;
        const mReal coeffAlpha = 0.5_mr * ka_ * alpha * (2 + alpha * (3 * a3_ + alpha * 4 * a4_));

        const mReal3 fArea = coeffAlpha * derArea;
        
        const mReal e0sq_A = dot(x12, x12) * area_inv;
        const mReal e1sq_A = dot(x13, x13) * area_inv;

        const mReal e0sq_A0 = eq.l0*eq.l0 * area0_inv;
        const mReal e1sq_A0 = eq.l1*eq.l1 * area0_inv;

        const mReal dotp = dot(x12, x13);

        const mReal dot_4A = 0.25_mr * eq.dotp * area0_inv;
        const mReal mixed_v = 0.125_mr * (e0sq_A0*e1sq_A + e1sq_A0*e0sq_A);
        const mReal beta = mixed_v - dot_4A * dotp * area_inv - 1.0_mr;

        const mReal3 derBeta = area_inv * ((0.25_mr * e1sq_A0 - dot_4A) * x12 +
                                          (0.25_mr * e0sq_A0 - dot_4A) * x13 +
                                          (dot_4A * dotp * area_inv - mixed_v) * derArea);
        
        const mReal3 derAlpha = area0_inv * derArea;
            
        const mReal coefAlpha = eq.a * mu_ * b1_ * beta;
        const mReal coefBeta  = eq.a * mu_ * (2*b2_*beta + alpha * b1_ + 1);

        const mReal3 fShear = coefAlpha * derAlpha + coefBeta * derBeta;

        return fArea + fShear;
    }
    
private:
    
    mReal ka_;
    mReal mu_;
    mReal a3_;
    mReal a4_;
    mReal b1_;
    mReal b2_;

    mReal length0_; ///< only useful when StressFree is false
    mReal area0_;   ///< only useful when StressFree is false
    mReal lscale_;
};

/// set type name
MIRHEO_TYPE_NAME(TriangleLimForce<StressFreeState::Active>, "TriangleLimForce<Active>");
/// set type name
MIRHEO_TYPE_NAME(TriangleLimForce<StressFreeState::Inactive>, "TriangleLimForce<Inactive>");

} // namespace mirheo
