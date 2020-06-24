// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "object_vector.h"

namespace mirheo
{

/** \brief Rod container.

    This is used to represent local or halo rods in RodVector.
    A rod is a chunk of particles connected implicitly in segments with additional 4 particles per edge.
    The number of particles per rod is then 5*n + 1 if n is the number of segments.
    Each object (called a rod) within a LocalRodVector has the same number of particles.
    Additionally to particle and object, data can be attached to each bisegment.
*/
class LocalRodVector : public LocalObjectVector
{
public:
    /** \brief Construct a LocalRodVector.
        \param [in] pv Parent RodVector
        \param [in] objSize Number of particles per object
        \param [in] nObjects Number of objects
    */
    LocalRodVector(ParticleVector *pv, int objSize, int nObjects = 0);
    virtual ~LocalRodVector();

    void resize(int np, cudaStream_t stream) override;
    void resize_anew(int np) override;

    /// get the number of segment per rod
    int getNumSegmentsPerRod() const;

public:
    DataManager dataPerBisegment; ///< contains bisegment data
};


/** \brief Rod objects container.

    Holds two LocalRodVector: local and halo.
 */
class RodVector: public ObjectVector
{
public:
    /** Construct a ObjectVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] nSegments Number of segments per rod
        \param [in] nObjects Number of rods
    */
    RodVector(const MirState *state, const std::string& name, real mass, int nSegments, int nObjects = 0);
    ~RodVector();

    /// get local LocalRodVector
    LocalRodVector* local() { return static_cast<LocalRodVector*>(ParticleVector::local()); }
    /// get halo LocalRodVector
    LocalRodVector* halo()  { return static_cast<LocalRodVector*>(ParticleVector::halo());  }
    /// get LocalRodVector from locality
    LocalRodVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }

    /** Add a new channel to hold additional data per bisegment.
        \tparam T The type of data to add
        \param [in] name channel name
        \param [in] persistence If the data should stich to the object or not when exchanging
        \param [in] shift If the data needs to be shifted when exchanged
    */
    template<typename T>
    void requireDataPerBisegment(const std::string& name, DataManager::PersistenceMode persistence,
                                 DataManager::ShiftMode shift = DataManager::ShiftMode::None)
    {
        _requireDataPerBisegment<T>(local(), name, persistence, shift);
        _requireDataPerBisegment<T>(halo(),  name, persistence, shift);
    }

private:
    template<typename T>
    void _requireDataPerBisegment(LocalRodVector *lrv, const std::string& name,
                                  DataManager::PersistenceMode persistence,
                                  DataManager::ShiftMode shift)
    {
        lrv->dataPerBisegment.createData<T> (name, lrv->getNumObjects());
        lrv->dataPerBisegment.setPersistenceMode(name, persistence);
        lrv->dataPerBisegment.setShiftMode(name, shift);
    }
};

} // namespace mirheo
