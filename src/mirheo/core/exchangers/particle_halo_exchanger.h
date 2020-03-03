#pragma once

#include "interface.h"

namespace mirheo
{

class ParticleVector;
class CellList;
class ParticlePacker;

/** \brief Pack and unpack data for halo particles exchange.

    The halo exchange consists in copying an image of all particles that are within one cut-off
    radius away to the neighbouring ranks.
    This leaves the original ParticleVector local data untouched.
    The result of this operation is stored in the halo LocalParticleVector.

    The halo exchange is accelerated by making use of the associated CellList of the ParticleVector.
 */
class ParticleHaloExchanger : public Exchanger
{
public:
    /// default constructor
    ParticleHaloExchanger();
    ~ParticleHaloExchanger();

    /** \brief Add a ParticleVector for halo exchange. 
        \param pv The ParticleVector to attach
        \param cl The associated cell-list of \p pv
        \param extraChannelNames The list of channels to exchange (additionally to the default positions and velocities)

        Multiple ParticleVector objects can be attached to the same halo exchanger.
     */
    void attach(ParticleVector *pv, CellList *cl, const std::vector<std::string>& extraChannelNames);

private:
    std::vector<CellList*> cellLists_;
    std::vector<ParticleVector*> particles_;
    std::vector<std::unique_ptr<ParticlePacker>> packers_, unpackers_;

    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
