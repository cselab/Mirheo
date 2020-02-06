#pragma once

#include "object_vector.h"

namespace mirheo
{

class LocalRodVector : public LocalObjectVector
{
public:
    LocalRodVector(ParticleVector *pv, int objSize, int nObjects = 0);
    virtual ~LocalRodVector();

    void resize(int np, cudaStream_t stream) override;
    void resize_anew(int np) override;

    int getNumSegmentsPerRod() const;

    DataManager dataPerBisegment;
};


class RodVector: public ObjectVector
{
public:
    RodVector(const MirState *state, const std::string& name, real mass, int nSegments, int nObjects = 0);
    ~RodVector();

    LocalRodVector* local() { return static_cast<LocalRodVector*>(ParticleVector::local()); }
    LocalRodVector* halo()  { return static_cast<LocalRodVector*>(ParticleVector::halo());  }
    LocalRodVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }

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
