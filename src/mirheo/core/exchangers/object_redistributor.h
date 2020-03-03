#pragma once

#include "interface.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

class ObjectVector;
class ObjectPacker;

/** \brief Pack and unpack data for object redistribution.

    As opposed to particles, objects must be redistributed as a whole for two reasons:
    - the particles of one object must stay into a contiguous chunk in memory
    - the objects might have associated data per object (or per bisegments for rods)

    The redistribution consists in moving (not copying) the object data from one rank to the other.
    It affects all objects that have left the current subdomain (an object belongs to a subdomain if its center of mass is inside).
 */
class ObjectRedistributor : public Exchanger
{
public:
    /// default constructor
    ObjectRedistributor();
    ~ObjectRedistributor();

    /** \brief Add an ObjectVector to the redistribution. 
        \param ov The ObjectVector to attach

        Multiple ObjectVector objects can be attached to the same redistribution object.
     */
    void attach(ObjectVector *ov);
    
private:
    std::vector<ObjectVector*> objects_;
    std::vector<std::unique_ptr<ObjectPacker>> packers_;
    
    void prepareSizes(size_t id, cudaStream_t stream) override;
    void prepareData (size_t id, cudaStream_t stream) override;
    void combineAndUploadData(size_t id, cudaStream_t stream) override;
    bool needExchange(size_t id) override;
};

} // namespace mirheo
