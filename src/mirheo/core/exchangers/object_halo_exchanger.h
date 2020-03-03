#pragma once

#include "interface.h"
#include "utils/map.h"

#include <mirheo/core/containers.h>

#include <memory>

namespace mirheo
{

class ObjectVector;
class ObjectPacker;
class MapEntry;

/** \brief Pack and unpack data for halo object exchange.

    The halo exchange consists in copying an image of all objects with bounding box that is within one cut-off
    radius away to the neighbouring ranks.
    This leaves the original ObjectVector local data untouched.
    The result of this operation is stored in the halo LocalObjectVector.

    This is needed only when the full object is needed on the neighbour ranks (e.g. \c Bouncer or ObjectBelongingChecker).
 */
class ObjectHaloExchanger : public Exchanger
{
public:
    /// default constructor
    ObjectHaloExchanger();
    ~ObjectHaloExchanger();

    /** \brief Add a ObjectVector for halo exchange. 
        \param ov The ObjectVector to attach
        \param rc The required cut-off radius
        \param extraChannelNames The list of channels to exchange (additionally to the default positions and velocities)

        Multiple ObjectVector objects can be attached to the same halo exchanger.
     */
    void attach(ObjectVector *ov, real rc, const std::vector<std::string>& extraChannelNames);

    PinnedBuffer<int>& getSendOffsets(size_t id); ///< \return send offset within the send buffer (in number of elements) of the given ov
    PinnedBuffer<int>& getRecvOffsets(size_t id); ///< \return recv offset within the send buffer (in number of elements) of the given ov
    DeviceBuffer<MapEntry>& getMap   (size_t id); ///< \return The map from LocalObjectVector to send buffer ids

private:
    std::vector<real> rcs_; ///< list of cut-off radius of all registered ovs
    std::vector<ObjectVector*> objects_; ///< list of registered ovs
    std::vector<std::unique_ptr<ObjectPacker>> packers_; ///< helper classes to pack the registered ovs
    std::vector<std::unique_ptr<ObjectPacker>> unpackers_; ///< helper classes to unpack the registered ovs
    std::vector<DeviceBuffer<MapEntry>> maps_; ///< maps from LocalObjectVector to send buffer ids

    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
