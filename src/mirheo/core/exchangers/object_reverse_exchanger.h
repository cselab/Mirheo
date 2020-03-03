#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

#include <vector>
#include <string>

namespace mirheo
{

class ObjectVector;
class ObjectHaloExchanger;
class ObjectPacker;

/** \brief Pack and unpack data from ghost particles back to the original bulk data.

    The ghost particles data must come from a ObjectHaloExchanger object.
    The attached ObjectVector objects must be the same as the ones in the external ObjectHaloExchanger 
    (and in the same order).
 */
class ObjectReverseExchanger : public Exchanger
{
public:
    /** \brief Construct a ObjectReverseExchanger
        \param entangledHaloExchanger The object that will create the ghost particles.
     */
    ObjectReverseExchanger(ObjectHaloExchanger *entangledHaloExchanger);
    virtual ~ObjectReverseExchanger();
    
    /** \brief Add an ObjectVector for reverse halo exchange. 
        \param ov The ObjectVector to attach
        \param channelNames The list of channels to send back
     */
    void attach(ObjectVector *ov, std::vector<std::string> channelNames);

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
