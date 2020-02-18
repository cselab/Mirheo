#pragma once

#include "generic_packer.h"

#include <mirheo/core/utils/cpu_gpu_defines.h>

#include <vector>

namespace mirheo
{

class LocalParticleVector;

/** \brief A packer specific to particle data only.

    The user can use the internal generic packer directly.
 */
struct ParticlePackerHandler
{
    /// The packer responsible for the particles
    GenericPackerHandler particles;

    /// Get the reuired size (in bytes) of the buffer to hold the packed data
    __D__ size_t getSizeBytes(int numElements) const
    {
        return particles.getSizeBytes(numElements);
    }
};

/// \brief Helper class to construct a \c ParticlePackerHandler.
class ParticlePacker
{
public:
    /** \brief Construct a \c ParticlePacker
        \param [in] predicate The channel filter that will be used to select the channels to be registered.
     */
    ParticlePacker(PackPredicate predicate);
    ~ParticlePacker();

    /** \brief Register the channels of a \c LocalParticleVector that meet the predicate requirements.
        \param [in] lpv The \c LocalParticleVector that holds the channels to be registered.
        \param [in] stream The stream used to transfer the channels information to the device.
     */
    virtual void update(LocalParticleVector *lpv, cudaStream_t stream);

    /// get a handler usable on device
    ParticlePackerHandler handler();

    /// Get the reuired size (in bytes) of the buffer to hold all the packed data
    virtual size_t getSizeBytes(int numElements) const;

protected:
    PackPredicate predicate_;    ///< filter to select the channels
    GenericPacker particleData_; ///< The packer used to pack and unpack the particle data
};

} // namespace mirheo
