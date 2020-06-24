// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "accumulators/force.h"
#include "fetchers.h"
#include "interface.h"
#include "parameters.h"

#include <mirheo/core/mirheo_state.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/rod_vector.h>

namespace mirheo
{

/** A GPU compatible functor that describes a filter for repulsive LJ interactions.
    This particular class allows interactions between all particles.
 */
class LJAwarenessNone
{
public:
    using ParamsType = LJAwarenessParamsNone; ///< Corresponding parameters type

    LJAwarenessNone() = default;
    /// Generic constructor
    LJAwarenessNone(__UNUSED const ParamsType& params) {}

    /// Setup internal state
    void setup(__UNUSED LocalParticleVector *lpv1, __UNUSED LocalParticleVector *lpv2) {}

    /// \return \c true if particles with ids \p srcId and \p dstId should interact, \c false otherwise
    __D__ inline bool interact(__UNUSED int srcId, __UNUSED int dstId) const {return true;}
};
/// set type name
MIRHEO_TYPE_NAME(LJAwarenessNone, "LJAwarenessNone");

/** A GPU compatible functor that describes a filter for repulsive LJ interactions.
    This particular class allows interactions only between particles of a different object.
 */
class LJAwarenessObject
{
public:
    using ParamsType = LJAwarenessParamsObject;  ///< Corresponding parameters type

    LJAwarenessObject() = default;
    /// Generic constructor
    LJAwarenessObject(__UNUSED const ParamsType& params) {}

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
/// set type name
MIRHEO_TYPE_NAME(LJAwarenessObject, "LJAwarenessObject");

/** A GPU compatible functor that describes a filter for repulsive LJ interactions.
    This particular class allows interactions only between particles of a different rod
    or particles within the same rod separated by a minimum number of segments.
    This is useful to avoid self crossing in rods.
 */
class LJAwarenessRod
{
public:
    using ParamsType = LJAwarenessParamsRod;  ///< Corresponding parameters type

    /// Constructor
    LJAwarenessRod(int minSegmentsDist) :
        minSegmentsDist_(minSegmentsDist)
    {}

    /// Generic constructor
    LJAwarenessRod(const ParamsType& params) :
        LJAwarenessRod(params.minSegmentsDist)
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

/// set type name
MIRHEO_TYPE_NAME(LJAwarenessRod, "LJAwarenessRod");


/** \brief Compute repulsive Lennard-Jones forces on the device
    \tparam Awareness A functor that describes which particles pairs interact
 */
template <class Awareness>
class PairwiseRepulsiveLJ : public PairwiseKernel, public ParticleFetcher
{
public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS // warnings in breathe
    using ViewType     = PVview;              ///< Compatible view type
    using ParticleType = Particle;            ///< Compatible particle type
    using HandlerType  = PairwiseRepulsiveLJ; ///< Corresponding handler
    using ParamsType   = RepulsiveLJParams;   ///< Corresponding parameters type
#endif // DOXYGEN_SHOULD_SKIP_THIS

    /// Constructor
    PairwiseRepulsiveLJ(real rc, real epsilon, real sigma, real maxForce, Awareness awareness) :
        ParticleFetcher(rc),
        sigma2_(sigma*sigma),
        maxForce_(maxForce),
        epsx24_sigma2_(24.0_r * epsilon / (sigma * sigma)),
        awareness_(awareness)
    {
        constexpr real sigmaFactor = 1.1224620483_r; // 2^(1/6)
        const real rm = sigmaFactor * sigma; // F(rm) = 0

        if (rm > rc)
        {
            const real maxSigma = rc / sigmaFactor;
            die("RepulsiveLJ: rm = %g > rc = %g; sigma must be lower than %g or rc must be larger than %g",
                rm, rc, maxSigma, rm);
        }
    }

    /// Generic constructor
    PairwiseRepulsiveLJ(real rc, const ParamsType& p, __UNUSED real dt, __UNUSED long seed=42424242) :
        PairwiseRepulsiveLJ{rc,
                            p.epsilon,
                            p.sigma,
                            p.maxForce,
                            mpark::get<typename Awareness::ParamsType>(p.varLJAwarenessParams)}
    {}

    /// Evaluate the force
    __D__ inline real3 operator()(ParticleType dst, int dstId, ParticleType src, int srcId) const
    {
        constexpr real tolerance = 1e-6_r;
        if (!awareness_.interact(src.i1, dst.i1))
            return make_real3(0.0_r);

        const real3 dr = dst.r - src.r;
        const real dr2 = dot(dr, dr);

        if (dr2 > rc2_ || dr2 < tolerance)
            return make_real3(0.0_r);

        const real rs2 = sigma2_ / dr2;
        const real rs4 = rs2*rs2;
        const real rs8 = rs4*rs4;
        const real rs14 = rs8*(rs4*rs2);

        const real IfI = epsx24_sigma2_ * (2*rs14 - rs8);

        return dr * math::min(math::max(IfI, 0.0_r), maxForce_);
    }

    /// initialize accumulator
    __D__ inline ForceAccumulator getZeroedAccumulator() const {return ForceAccumulator();}

    /// get the handler that can be used on device
    const HandlerType& handler() const
    {
        return (const HandlerType&) (*this);
    }

    void setup(LocalParticleVector *lpv1, LocalParticleVector *lpv2,
               __UNUSED CellList *cl1, __UNUSED CellList *cl2, __UNUSED const MirState *state) override
    {
        awareness_.setup(lpv1, lpv2);
    }

    /// \return type name string
    static std::string getTypeName()
    {
        return constructTypeName<Awareness>("PairwiseRepulsiveLJ");
    }

private:
    real sigma2_, maxForce_;
    real epsx24_sigma2_;

    Awareness awareness_;
};

} // namespace mirheo
