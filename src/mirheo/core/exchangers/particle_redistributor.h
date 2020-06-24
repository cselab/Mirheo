#pragma once

#include "interface.h"

namespace mirheo
{

class ParticleVector;
class CellList;
class ParticlePacker;

/** \brief Pack and unpack data for particle redistribution.

    The redistribution consists in moving (not copying) the particles from one rank to the other.
    It affects all particles that have left the current subdomain.
    The redistribution is accelerated by making use of the cell-lists of the ParticleVector.
    This allows to check only the particles that are on the boundary cells;
    However, this assumes that only those particles leave the domain.
 */
class ParticleRedistributor : public Exchanger
{
public:
    /// default constructor
    ParticleRedistributor();
    ~ParticleRedistributor();

    /** \brief Add a ParticleVector to the redistribution.
        \param pv The ParticleVector to attach
        \param cl The associated cell-list of \p pv.

        Multiple ParticleVector objects can be attached to the same redistribution object.
     */
    void attach(ParticleVector *pv, CellList *cl);

private:
    std::vector<ParticleVector*> particles_;
    std::vector<CellList*> cellLists_;
    std::vector<std::unique_ptr<ParticlePacker>> packers_;

    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
