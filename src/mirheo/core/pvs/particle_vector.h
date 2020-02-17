#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/mirheo_object.h>
#include <mirheo/core/pvs/data_manager.h>
#include <mirheo/core/utils/pytypes.h>

#include <memory>
#include <string>
#include <vector>

namespace mirheo
{

class ParticleVector;

/// Designs local or halo data
enum class ParticleVectorLocality
{
    Local,
    Halo
};

/** Transform a \c ParticleVectorLocality into a string.
    \param [in] locality Data locality
    \return a string that describes the locality
*/
std::string getParticleVectorLocalityStr(ParticleVectorLocality locality);

/** \brief Particles container. 
    This is used to represent local or halo particles in \c ParticleVector.
*/
class LocalParticleVector
{
public:
    /** Construct a \c LocalParticleVector.
        \param [in] pv Pointer to the parent \c ParticleVector.
        \param [in] np Number of particles.
     */
    LocalParticleVector(ParticleVector *pv, int np = 0);
    virtual ~LocalParticleVector();

    /// swap two \c LocalParticleVector
    friend void swap(LocalParticleVector &, LocalParticleVector &);
    template <typename T>
    friend void swap(LocalParticleVector &, T &) = delete;  // Disallow implicit upcasts.

    /// return the number of particles
    int size() const noexcept { return np_; }

    /** resize the container, preserving the data.
        \param [in] n new number of particles
        \param [in] stream that is used to copy data
    */ 
    virtual void resize(int n, cudaStream_t stream);

    /** resize the container, without preserving the data.
        \param [in] n new number of particles
    */ 
    virtual void resize_anew(int n);    

    /// get forces container reference
    PinnedBuffer<Force>& forces();
    /// get positions container reference
    PinnedBuffer<real4>& positions();
    /// get velocities container reference
    PinnedBuffer<real4>& velocities();

    /** \brief Set a unique Id for each particle in the simulation.
        \param [in] comm MPI communicator of the simulation
        \param [in] stream Stream used to transfer data between host and device
        
        The ids are stored in the channel ChannelNames::globalIds.
     */
    virtual void computeGlobalIds(MPI_Comm comm, cudaStream_t stream);

    /// get parent \c ParticleVector
    ParticleVector* parent() {return pv_;}
    /// get parent \c ParticleVector
    const ParticleVector* parent() const {return pv_;}
    
public:
    DataManager dataPerParticle; ///< Contains all particle channels

private:
    ParticleVector *pv_; ///< parent \c ParticleVector
    int np_; ///< number of particles
};

/** \brief Base particles container.

    Holds two \c LocalParticleVector: local and halo.
    The local one contains the data present in the current subdomain.
    The halo one is used to exchange particle data with the neighboring ranks.

    By default, contains positions, velocities, forces and global ids.
 */
class ParticleVector : public MirSimulationObject
{
public:
    /** Construct a \c ParticleVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] n Number of particles
    */
    ParticleVector(const MirState *state, const std::string& name, real mass, int n = 0);
    ~ParticleVector() override;

    /// get the local \c LocalParticleVector 
    LocalParticleVector* local() { return local_.get(); }
    /// get the halo \c LocalParticleVector 
    LocalParticleVector* halo()  { return halo_.get();  }

    /** get the \c LocalParticleVector corresponding to a given locality
        \param [in] locality local or halo
     */ 
    LocalParticleVector* get(ParticleVectorLocality locality)
    {
        return (locality == ParticleVectorLocality::Local) ? local() : halo();
    }

    /// get the local \c LocalParticleVector 
    const LocalParticleVector* local() const { return local_.get(); }
    /// get the halo \c LocalParticleVector 
    const LocalParticleVector* halo()  const { return  halo_.get(); }

    void checkpoint(MPI_Comm comm, const std::string& path, int checkpointId) override;
    void restart   (MPI_Comm comm, const std::string& path) override;

    
    /** Python getters / setters
        Use default blocking stream
    */
    /// \{
    std::vector<int64_t> getIndices_vector();
    PyTypes::VectorOfReal3 getCoordinates_vector();
    PyTypes::VectorOfReal3 getVelocities_vector();
    PyTypes::VectorOfReal3 getForces_vector();
    
    void setCoordinates_vector(const std::vector<real3>& coordinates);
    void setVelocities_vector(const std::vector<real3>& velocities);
    void setForces_vector(const std::vector<real3>& forces);
    ///\}

    /** Add a new channel to hold additional data per particle.
        \tparam T The type of data to add
        \param [in] name channel name
        \param [in] persistence If the data should stich to the particles or not when exchanged
        \param [in] shift If the data needs to be shifted when exchanged
     */
    template<typename T>
    void requireDataPerParticle(const std::string& name, DataManager::PersistenceMode persistence,
                                DataManager::ShiftMode shift = DataManager::ShiftMode::None)
    {
        _requireDataPerParticle<T>(local(), name, persistence, shift);
        _requireDataPerParticle<T>(halo(),  name, persistence, shift);
    }

    /// get the particle mass
    real getMassPerParticle() const;
    
protected:
    /** Construct a \c ParticleVector
        \param [in] state The simulation state
        \param [in] name Name of the pv
        \param [in] mass Mass of one particle
        \param [in] local Local particles
        \param [in] halo Halo particles
    */
    ParticleVector(const MirState *state, const std::string& name, real mass,
                   std::unique_ptr<LocalParticleVector>&& local,
                   std::unique_ptr<LocalParticleVector>&& halo );

    /// Exchange map used when reading a file in MPI
    using ExchMap = std::vector<int>;
    
    /// Simple helper structure
    struct ExchMapSize
    {
        ExchMap map; ///< echange map
        int newSize; ///< size after exchange
    };

    /** Dump particle data into a file
        \param [in] comm MPI carthesian comm used to perform I/O and exchange data across ranks
        \param [in] path Destination folder
        \param [in] checkpointId The Id of the dump  
     */
    virtual void _checkpointParticleData(MPI_Comm comm, const std::string& path, int checkpointId);

    /** Load particle data from a file
        \param [in] comm MPI carthesian comm used to perform I/O and exchange data across ranks
        \param [in] path Source folder that contains the file
        \param [in] chunkSize Every chunk of this number of particles will always stay together. 
                              This is useful for \c ObjectVector.
        \return Exchange map that is used to redistribute the chunks of data across ranks.
     */
    virtual ExchMapSize _restartParticleData(MPI_Comm comm, const std::string& path, int chunkSize);

private:
    template<typename T>
    void _requireDataPerParticle(LocalParticleVector *lpv, const std::string& name,
                                 DataManager::PersistenceMode persistence,
                                 DataManager::ShiftMode shift)
    {
        lpv->dataPerParticle.createData<T> (name, lpv->size());
        lpv->dataPerParticle.setPersistenceMode(name, persistence);
        lpv->dataPerParticle.setShiftMode(name, shift);
    }

public:    
    bool haloValid   {false}; ///< true if the halo is up to date
    bool redistValid {false}; ///< true if the particles are redistributed

    int cellListStamp {0}; ///< stamp that keep track if the cell list is up to date

private:
    real mass_;
    std::unique_ptr<LocalParticleVector> local_, halo_;
};

} // namespace mirheo
