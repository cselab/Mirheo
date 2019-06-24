#pragma once

#include "object_vector.h"

#include <core/containers.h>
#include <core/datatypes.h>

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
    RodVector(const YmrState *state, std::string name, float mass, int nSegments, int nObjects = 0);
    ~RodVector();

    LocalRodVector* local() { return static_cast<LocalRodVector*>(ParticleVector::local()); }
    LocalRodVector* halo()  { return static_cast<LocalRodVector*>(ParticleVector::halo());  }


    template<typename T>
    void requireDataPerBisegment(std::string name, DataManager::PersistenceMode persistence)
    {
        requireDataPerBisegment<T>(name, persistence, 0);
    }

    template<typename T>
    void requireDataPerBisegment(std::string name, DataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        requireDataPerBisegment<T>(local(), name, persistence, shiftDataSize);
        requireDataPerBisegment<T>(halo(),  name, persistence, shiftDataSize);
    }

private:
    template<typename T>
    void requireDataPerBisegment(LocalRodVector *lrv, std::string name, DataManager::PersistenceMode persistence, size_t shiftDataSize)
    {
        lrv->dataPerBisegment.createData<T> (name, lrv->nObjects);
        lrv->dataPerBisegment.setPersistenceMode(name, persistence);
        if (shiftDataSize != 0) lrv->dataPerBisegment.requireShift(name, shiftDataSize);
    }

};
