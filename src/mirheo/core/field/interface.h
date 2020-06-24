#pragma once

#include <mirheo/core/domain.h>
#include <mirheo/core/containers.h>
#include <mirheo/core/mirheo_object.h>

#include <mirheo/core/utils/cpu_gpu_defines.h>
#include <mirheo/core/utils/helper_math.h>

#include <vector>

namespace mirheo
{

#ifndef __NVCC__
/// default tex3D so that things compile on the host; does nothing
template<typename T>
T tex3D(__UNUSED cudaTextureObject_t t,
        __UNUSED real x, __UNUSED real y, __UNUSED real z)
{
    return T();
}
#endif

/** \brief a device-compatible structure that represents a scalar field
 */
class FieldDeviceHandler
{
public:
    /** \brief Evaluate the field at a given position
        \param [in] x The position, in local coordinates
        \return The scalar value at \p x

        \rst
        .. warning::
           The position must be inside the subdomain enlarged with a given margin (see \c Field)

        \endrst
     */
    __D__ inline real operator()(real3 x) const
    {
        //https://en.wikipedia.org/wiki/Trilinear_interpolation
        real s000, s001, s010, s011, s100, s101, s110, s111;
        real sx00, sx01, sx10, sx11, sxy0, sxy1, sxyz;

        const real3 texcoord = math::floor((x + extendedDomainSize_*0.5_r) * invh_);
        const real3 lambda = (x - (texcoord * h_ - extendedDomainSize_*0.5_r)) * invh_;

        auto access = [this, &texcoord] (int dx, int dy, int dz)
        {
            const auto val = tex3D<float>(fieldTex_,
                                          static_cast<float>(texcoord.x + static_cast<real>(dx)),
                                          static_cast<float>(texcoord.y + static_cast<real>(dy)),
                                          static_cast<float>(texcoord.z + static_cast<real>(dz)));
            return static_cast<real>(val);
        };

        s000 = access(0, 0, 0);
        s001 = access(0, 0, 1);
        s010 = access(0, 1, 0);
        s011 = access(0, 1, 1);

        s100 = access(1, 0, 0);
        s101 = access(1, 0, 1);
        s110 = access(1, 1, 0);
        s111 = access(1, 1, 1);

        sx00 = s000 * (1 - lambda.x) + lambda.x * s100;
        sx01 = s001 * (1 - lambda.x) + lambda.x * s101;
        sx10 = s010 * (1 - lambda.x) + lambda.x * s110;
        sx11 = s011 * (1 - lambda.x) + lambda.x * s111;

        sxy0 = sx00 * (1 - lambda.y) + lambda.y * sx10;
        sxy1 = sx01 * (1 - lambda.y) + lambda.y * sx11;

        sxyz = sxy0 * (1 - lambda.z) + lambda.z * sxy1;

        return sxyz;
    }

protected:
    cudaTextureObject_t fieldTex_; ///< Texture object that points to the uniform grid field data
    real3 h_;    ///< grid spacing
    real3 invh_; ///< 1 / h
    real3 extendedDomainSize_; ///< subdomain size extended with a margin
};

/** \brief Driver class used to create a FieldDeviceHandler.
 */
class Field : public FieldDeviceHandler, public MirSimulationObject
{
public:
    /** \brief Construct a \c Field object
        \param [in] state The global state of the system
        \param [in] name The name of the field object
        \param [in] h the grid size
     */
    Field(const MirState *state, std::string name, real3 h);
    virtual ~Field();

    /// move constructor
    Field(Field&&);

    /// \return The handler that can be used on the device
    const FieldDeviceHandler& handler() const;

    /** Prepare the internal state of the \c Field.
        Must be called before handler().
        \param [in] comm The cartesian communicator of the domain.
     */
    virtual void setup(const MPI_Comm& comm) = 0;

protected:
    int3 resolution_;       ///< number of grid points along each dimension
    cudaArray *fieldArray_; ///< contains the field data

    /// Additional distance along each direction in which to store the field data around the local subdomain.
    /// This is used e.g. to avoid communicating "ghost walls" when ObjectVector objects interact with the walls.
    const real3 margin3_{5, 5, 5};

    /** \brief copy the given grid data to the internal buffer and create the associated texture object
        \param [in] fieldDevPtr The scalar values at each grid point (x is the fast index)
    */
    void _setupArrayTexture(const float *fieldDevPtr);
};

} // namespace mirheo
