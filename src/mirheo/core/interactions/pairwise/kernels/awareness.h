// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "parameters.h"

#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

/** A GPU compatible functor that describes a filter for pairwise interactions.
    This particular class allows interactions between all particles.
 */
class AwarenessNone
{
public:
    using ParamsType = AwarenessParamsNone; ///< Corresponding parameters type

    AwarenessNone() = default;
    /// Generic constructor
    AwarenessNone(__UNUSED const ParamsType& params) {}

    /// Setup internal state
    void setup(__UNUSED LocalParticleVector *lpv1, __UNUSED LocalParticleVector *lpv2) {}

    /// \return \c true if particles with ids \p srcId and \p dstId should interact, \c false otherwise
    __D__ inline bool interact(__UNUSED int srcId, __UNUSED int dstId) const {return true;}
};


/** A GPU compatible functor that describes a filter for pairwise interactions.
    This particular class allows interactions only between particles of a different object.
 */
class AwarenessObject
{
public:
    using ParamsType = AwarenessParamsObject;  ///< Corresponding parameters type

    AwarenessObject() = default;
    /// Generic constructor
    AwarenessObject(__UNUSED const ParamsType& params) {}

    /// Setup internal state
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2)
    {
        auto ov1 = dynamic_cast<ObjectVector*>(lpv1->parent());
        auto ov2 = dynamic_cast<ObjectVector*>(lpv2->parent());

        self_ = false;
        if (ov1 != nullptr && ov2 != nullptr && lpv1 == lpv2)
        {
            self_ = true;
            objSize_ = ov1->getObjectSize();
        }
    }

    /// \return \c true if particles with ids \p srcId and \p dstId should interact, \c false otherwise
    __D__ inline bool interact(int srcId, int dstId) const
    {
        if (self_)
        {
            const int dstObjId = dstId / objSize_;
            const int srcObjId = srcId / objSize_;

            if (dstObjId == srcObjId)
                return false;
        }
        return true;
    }

private:
    bool self_ {false};
    int objSize_ {0};
};

/** A GPU compatible functor that describes a filter for pairwise interactions.
    This particular class allows interactions only between particles of a different rod
    or particles within the same rod separated by a minimum number of segments.
    This is useful to avoid self crossing in rods.
 */
class AwarenessRod
{
public:
    using ParamsType = AwarenessParamsRod;  ///< Corresponding parameters type

    /// Constructor
    AwarenessRod(int minSegmentsDist) :
        minSegmentsDist_(minSegmentsDist)
    {}

    /// Generic constructor
    AwarenessRod(const ParamsType& params) :
        AwarenessRod(params.minSegmentsDist)
    {}

    /// Setup internal state
    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2)
    {
        auto rv1 = dynamic_cast<RodVector*>(lpv1->parent());
        auto rv2 = dynamic_cast<RodVector*>(lpv2->parent());

        self_ = false;
        if (rv1 != nullptr && rv2 != nullptr && lpv1 == lpv2)
        {
            self_ = true;
            objSize_ = rv1->getObjectSize();
        }
    }

    /// \return \c true if particles with ids \p srcId and \p dstId should interact, \c false otherwise
    __D__ inline bool interact(int srcId, int dstId) const
    {
        if (self_)
        {
            const int dstObjId = dstId / objSize_;
            const int srcObjId = srcId / objSize_;

            if (dstObjId == srcObjId)
            {
                const int srcSegId = (dstId % objSize_) / 5;
                const int dstSegId = (srcId % objSize_) / 5;

                if (math::abs(srcSegId - dstSegId) <= minSegmentsDist_)
                    return false;
            }
        }
        return true;
    }

private:
    bool self_ {false};
    int objSize_ {0};
    int minSegmentsDist_{0};
};


} // namespace mirheo
