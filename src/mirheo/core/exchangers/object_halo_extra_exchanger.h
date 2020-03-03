#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

#include <memory>
#include <vector>
#include <string>

namespace mirheo
{

class ObjectVector;
class ObjectPacker;
class ObjectHaloExchanger;

/** \brief Pack and unpack extra data for halo object exchange.
    \see ObjectHaloExchanger

    This class only exchanges the additional data (not e.g. the default particle's positions and velocities).
    It uses the packing map from an external ObjectHaloExchanger. 
    The attached ObjectVector objects must be the same as the ones in the external ObjectHaloExchanger 
    (and in the same order).
 */
class ObjectExtraExchanger : public Exchanger
{
public:
    /** \brief Construct a ObjectExtraExchanger
        \param entangledHaloExchanger The object that will contain the packing maps.
     */
    ObjectExtraExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    ~ObjectExtraExchanger();

    /** \brief Add a ObjectVector for halo exchange. 
        \param ov The ObjectVector to attach
        \param extraChannelNames The list of channels to exchange
     */
    void attach(ObjectVector *ov, const std::vector<std::string>& extraChannelNames);

private:
    std::vector<ObjectVector*> objects_;
    ObjectHaloExchanger *entangledHaloExchanger_;
    std::vector<std::unique_ptr<ObjectPacker>> packers_, unpackers_;
    
    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
