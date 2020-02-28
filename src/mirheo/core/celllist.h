#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo
{
/// describes if the position should be projected inside the
/// local subdomain or not
enum class CellListsProjection
{
    Clamp, NoClamp
};

/** A device-compatible structure that represents the cell-lists structure.
    Contains geometric info (number of cells, cell sizes) and associated 
    particles info (number of particles per cell and cell-starts).
 */
class CellListInfo
{
public:
    /** \brief Construct a CellListInfo object
        \param [in] h The size of a single cell along each dimension
        \param [in] localDomainSize Size of the local domain
        
        This will create a cell-lists structure with cell sizes which are larger
        or equal to \p h, such that the number of cells fit exactly inside the 
        local domain size.
     */
    CellListInfo(real3 h, real3 localDomainSize);

    /** \brief Construct a CellListInfo object
        \param [in] rc The minimum size of a single cell along every dimension
        \param [in] localDomainSize Size of the local domain
        
        This will create a cell-lists structure with cell sizes which are larger
        or equal to \p rc, such that the number of cells fit exactly inside the 
        local domain size.
     */
    CellListInfo(real rc, real3 localDomainSize);

#ifdef __CUDACC__
    /** \brief map 3D cell indices to linear cell index.
        \param [in] ix Cell index in the x direction
        \param [in] iy Cell index in the y direction
        \param [in] iz Cell index in the z direction
        \return Linear cell index
     */
    __device__ __host__ inline int encode(int ix, int iy, int iz) const
    {
        return (iz*ncells.y + iy)*ncells.x + ix;
    }

    /** \brief map linear cell index to 3D cell indices.
        \param [in] cid Linear cell index
        \param [out] ix Cell index in the x direction
        \param [out] iy Cell index in the y direction
        \param [out] iz Cell index in the z direction
     */
    __device__ __host__ inline void decode(int cid, int& ix, int& iy, int& iz) const
    {
        ix = cid % ncells.x;
        iy = (cid / ncells.x) % ncells.y;
        iz = cid / (ncells.x * ncells.y);
    }

    /// see encode()
    __device__ __host__ inline int encode(int3 cid3) const
    {
        return encode(cid3.x, cid3.y, cid3.z);
    }

    /// see decode()
    __device__ __host__ inline int3 decode(int cid) const
    {
        int3 res;
        decode(cid, res.x, res.y, res.z);
        return res;
    }

    /** \brief Map from position to cell indices
        \tparam Projection if the cell indices must be clamped or not
        \param [in] x The position in **local coordinates**
        \return cell indices
     */
    template<CellListsProjection Projection = CellListsProjection::Clamp>
    __device__ __host__ inline int3 getCellIdAlongAxes(const real3 x) const
    {
        const int3 v = make_int3( math::floor(invh_ * (x + 0.5_r * localDomainSize)) );

        if (Projection == CellListsProjection::Clamp)
            return math::min( ncells - 1, math::max(make_int3(0), v) );
        else
            return v;
    }

    /** \brief Map from position to linear indices
        \tparam Projection if the cell indices must be clamped or not
        \tparam T The vector type that represents the position
        \param [in] x The position in **local coordinates**
        \return linear cell index

        \rst
        .. warning:: 
           If The projection is set to CellListsProjection::NoClamp, 
           this function will return -1 if the particle is outside the subdomain.
        \endrst
     */
    template<CellListsProjection Projection = CellListsProjection::Clamp, typename T>
    __device__ __host__ inline int getCellId(const T x) const
    {
        const int3 cid3 = getCellIdAlongAxes<Projection>(make_real3(x));

        if (Projection == CellListsProjection::NoClamp)
        {
            if (cid3.x < 0 || cid3.x >= ncells.x ||
                cid3.y < 0 || cid3.y >= ncells.y ||
                cid3.z < 0 || cid3.z >= ncells.z)
                return -1;
        }

        return encode(cid3);
    }
#endif

public:
    int3 ncells;   ///< Number of cells along each direction in the local domain
    int  totcells; ///< total number of cells in the local domain
    real3 localDomainSize; ///< dimensions of the subdomain
    real rc;    ///< cutoff radius
    real3 h;    ///< dimensions of the cells along each direction
    
    int *cellSizes;  ///< number of particles contained in each cell
    int *cellStarts; ///< exclusive prefix sum of cellSizes

    /// used to reorder particles when building the cell lists:
    /// \c order[pid] is the destination index of the particle with index \c pid before reordering
    int *order;

private:
    real3 invh_; ///< 1 / h
};


/** \brief Contains the cell-list data for a given ParticleVector. 

    As opposed to the PrimaryCellList class, it contains a **copy** of the 
    attached ParticleVector.
    This means that the original ParticleVector data will not be reorder but rather copied into this container.
    This is useful when several CellList object can be attached to the same ParticleVector or if the ParticleVector
    must not be reordered such as e.g. for ObjectVector objects.
 */
class CellList : public CellListInfo
{
public:    
    /** Construct a CellList object
        \param [in] pv The ParticleVector to attach.
        \param [in] rc The maximum cut-off radius that can be used with that cell list.
        \param [in] localDomainSize The size of the local subdomain
     */
    CellList(ParticleVector *pv, real rc, real3 localDomainSize);

    /** Construct a CellList object
        \param [in] pv The ParticleVector to attach.
        \param [in] resolution The number of cells along each dimension
        \param [in] localDomainSize The size of the local subdomain
     */
    CellList(ParticleVector *pv, int3 resolution, real3 localDomainSize);

    virtual ~CellList();

    /// \return the device-compatible handler
    CellListInfo cellInfo();

    /** \brief construct the cell-list associated with the attached ParticleVector
        \param [in] stream The stream used to execute the process
     */
    virtual void build(cudaStream_t stream);

    /** \brief Accumulate the channels from the data contained inside the cell-list 
        container to the attached ParticleVector.
        \param [in] channelNames List that contains the names of all the channels to accumulate
        \param [in] stream Execution stream
     */
    virtual void accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream);

    /** \brief Copy the channels from the attached ParticleVector to the cell-lists data.
        \param [in] channelNames List that contains the names of all the channels to copy
        \param [in] stream Execution stream
     */
    virtual void gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream);

    /** \brief Clear channels contained inside the cell-list
        \param [in] channelNames List that contains the names of all the channels to clear
        \param [in] stream Execution stream
    */
    void clearChannels(const std::vector<std::string>& channelNames, cudaStream_t stream);

    
    /** \brief Create a view that points to the data contained in the cell-lists
        \tparam ViewType The type of the view to create
        \return View that points to the cell-lists data
     */    
    template <typename ViewType>
    ViewType getView() const
    {
        return ViewType(pv_, localPV_);
    }

    /** \brief Add an extra channel to the cell-list.
        \tparam T The type of data to add
        \param [in] name Name of the channel
     */
    template <typename T>
    void requireExtraDataPerParticle(const std::string& name)
    {
        particlesDataContainer_->dataPerParticle.createData<T>(name);
    }

    /// \return The LocalParticleVector that contains the data in the cell-list
    LocalParticleVector* getLocalParticleVector();
    
protected:
    /// initialize internal buffers; used in the constructor
    void _initialize();
    /// \return \c true if needs to build the cell-lists
    bool _checkNeedBuild() const;
    /// add needed channels to the internal data container so it matches
    /// the attached ParricleVector  
    void _updateExtraDataChannels(cudaStream_t stream);
    /// Compute the number of particles per cell
    void _computeCellSizes(cudaStream_t stream);
    /// Compute the cell starts; requires cell sizes
    void _computeCellStarts(cudaStream_t stream);
    /// reorder the positions and create \c order. requires cell starts
    void _reorderPositionsAndCreateMap(cudaStream_t stream);
    /// reorder the rest of the data that is persistent; requires \c order
    void _reorderPersistentData(cudaStream_t stream);

    /// build cell lists (uses the above functions)
    void _build(cudaStream_t stream);

    /// see accumulateChannels(); for one channel.
    void _accumulateExtraData(const std::string& channelName, cudaStream_t stream);

    /** reorder a given channel according to the internal map
        \param [in] channelName The name of the channel to reorder
        \param [in,out] channelDesc the channel data 
        \param [in] stream The execution stream
    */
    void _reorderExtraDataEntry(const std::string& channelName,
                                const DataManager::ChannelDescription *channelDesc,
                                cudaStream_t stream);

    /** \return a name specific to that cell list, its attached ParticleVector and
        cut-off radius; useful for logging
    */
    virtual std::string _makeName() const;

protected:
    int changedStamp_{-1}; ///< Helper to keep track of the validity of the cell-list

    DeviceBuffer<char> scanBuffer; ///< work space to perform the prefix sum
    DeviceBuffer<int> cellStarts; ///< Container of the cell starts
    DeviceBuffer<int> cellSizes; ///< Container of the cell sizes
    DeviceBuffer<int> order; ///< container of the reorder map

    std::unique_ptr<LocalParticleVector> particlesDataContainer_; ///< local data that holds reordered copy of the attached particle data
    LocalParticleVector *localPV_; ///< will point to particlesDataContainer or pv->local() if Primary
    
    ParticleVector *pv_; ///< The attached ParticleVector
};

/** \brief Contains the cell-list map for a given ParticleVector. 

    As opposed to the CellList class, the data is stored only in the ParticleVector.
    This means that the original ParticleVector data will be reorder according to this cell-list.
    This allows to save memory and avoid extra copies. 
    On the other hand, this class must not be used with ObjectVector objects.
 */
class PrimaryCellList : public CellList
{
public:
    /** Construct a PrimaryCellList object
        \param [in] pv The ParticleVector to attach.
        \param [in] rc The maximum cut-off radius that can be used with that cell list.
        \param [in] localDomainSize The size of the local subdomain
     */
    PrimaryCellList(ParticleVector *pv, real rc, real3 localDomainSize);

    /** Construct a PrimaryCellList object
        \param [in] pv The ParticleVector to attach.
        \param [in] resolution The number of cells along each dimension
        \param [in] localDomainSize The size of the local subdomain
    */
    PrimaryCellList(ParticleVector *pv, int3 resolution, real3 localDomainSize);

    ~PrimaryCellList();
    
    void build(cudaStream_t stream);

    void accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream) override;
    void gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream) override;

protected:
    /// swap data between the internal container with the attached particle data
    void _swapPersistentExtraData();
    std::string _makeName() const override;
};

} // namespace mirheo
