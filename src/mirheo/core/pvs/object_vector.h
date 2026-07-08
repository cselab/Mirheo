// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "particle_vector.h"

#include <mirheo/core/containers.h>
#include <mirheo/core/mesh/mesh.h>
#include <mirheo/core/utils/common.h>

namespace mirheo
{

/** \brief Objects container.

    This is used to represent local or halo objects in ObjectVector.
    An object is a chunk of particles, each chunk with the same number of particles within an ObjectVector.
    Additionally, data can be attached to each of those chunks.
*/
class LocalObjectVector: public LocalParticleVector
{
public:
    /** \brief Construct a LocalParticleVector.
        \param [in] pv Parent ObjectVector
        \param [in] objSize Number of particles per object
        \param [in] nObjects Number of objects
    */
    LocalObjectVector(ParticleVector *pv, int objSize, int nObjects = 0);
    virtual ~LocalObjectVector();

    /// swap two LocalObjectVector
    friend void swap(LocalObjectVector &, LocalObjectVector &);
    template <typename T>
    friend void swap(LocalObjectVector &, T &) = delete;  // Disallow implicit upcasts.

    void resize(int np, cudaStream_t stream) override;
    void resize_anew(int np) override;

    void computeGlobalIds(MPI_Comm comm, cudaStream_t stream) override;

    /// get positions of the mesh vertices
    virtual PinnedBuffer<real4>* getMeshVertices(cudaStream_t stream);
    /// get positions of the old mesh vertices
    virtual PinnedBuffer<real4>* getOldMeshVertices(cudaStream_t stream);
    /// get forces on the mesh vertices
    virtual PinnedBuffer<Force>* getMeshForces(cudaStream_t stream);

    /// get number of particles per object
    int getObjectSize() const;
    /// get number of objects
    int getNumObjects() const;

private:
    int _computeNobjects(int np) const;

public:
    DataManager dataPerObject; ///< contains object data

private:
    int objSize_;
    int nObjects_;
};


/** \brief Base objects container.

    Holds two LocalObjectVector: local and halo.
 */
class ObjectVector : public ParticleVector
{
public:
    /** Construct a ObjectVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] objSize Number of particles per object
        \param [in] nObjects Number of objects
    */
    ObjectVector(const MirState *state, const std::string& name, real mass, int objSize, int nObjects = 0);
    virtual ~ObjectVector();

    /** \brief Compute Extents and center of mass of each object in the given LocalObjectVector
        \param [in] stream The stream to execute the kernel on.
        \param [in] locality Specify which LocalObjectVector to compute the data
    */
    void findExtentAndCOM(cudaStream_t stream, ParticleVectorLocality locality);

    /// get local LocalObjectVector
    LocalObjectVector* local() { return static_cast<LocalObjectVector*>(ParticleVector::local()); }
    /// get halo LocalObjectVector
    LocalObjectVector* halo()  { return static_cast<LocalObjectVector*>(ParticleVector::halo());  }
    /// get LocalObjectVector from locality
    LocalObjectVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }


    void checkpoint (MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart    (MPI_Comm comm, const std::string& path) override;


    /** Add a new channel to hold additional data per object.
        \tparam T The type of data to add
        \param [in] name channel name
        \param [in] persistence If the data should stich to the objects or not when exchanging
        \param [in] shift If the data needs to be shifted when exchanged
    */
    template<typename T>
    void requireDataPerObject(const std::string& name, DataManager::PersistenceMode persistence,
                              DataManager::ShiftMode shift = DataManager::ShiftMode::None)
    {
        _requireDataPerObject<T>(local(), name, persistence, shift);
        _requireDataPerObject<T>(halo(),  name, persistence, shift);
    }

    /// get number of particles per object
    int getObjectSize() const;

protected:
    /** Construct a ObjectVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] objSize Number of particles per object
        \param [in] local Local LocalObjectVector
        \param [in] halo Halo LocalObjectVector
    */
    ObjectVector(const MirState *state, const std::string& name, real mass, int objSize,
                 std::unique_ptr<LocalParticleVector>&& local,
                 std::unique_ptr<LocalParticleVector>&& halo);

    /** Dump object data into a file
        \param [in] comm MPI Cartesian comm used to perform I/O and exchange data across ranks
        \param [in] path Destination folder
        \param [in] checkpointId The Id of the dump
     */
    virtual void _checkpointObjectData(MPI_Comm comm, const std::string& path, int checkpointId);

    /** Load object data from a file
        \param [in] comm MPI Cartesian comm used to perform I/O and exchange data across ranks
        \param [in] path Source folder that contains the file
        \param [in] ms Map to exchange the object data accross ranks, computed from _restartParticleData()
    */
    virtual void _restartObjectData(MPI_Comm comm, const std::string& path, const ExchMapSize& ms);

private:
    void _snapshotObjectData(MPI_Comm comm, const std::string& filename);

    template<typename T>
    void _requireDataPerObject(LocalObjectVector* lov, const std::string& name,
                               DataManager::PersistenceMode persistence,
                               DataManager::ShiftMode shift)
    {
        lov->dataPerObject.createData<T> (name, lov->getNumObjects());
        lov->dataPerObject.setPersistenceMode(name, persistence);
        lov->dataPerObject.setShiftMode(name, shift);
    }

public:
    std::shared_ptr<Mesh> mesh; ///< Triangle mesh that can be used to represent the object surface.

private:
    int objSize_;
};

} // namespace mirheo
