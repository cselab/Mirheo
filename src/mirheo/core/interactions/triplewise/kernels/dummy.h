// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

//maybe fix
#include <mirheo/core/interactions/pairwise/kernels/fetchers.h>
#include <array>

#include "interface.h"
#include "parameters.h"

namespace mirheo
{

class TriplewiseDummyHandler : public ParticleFetcher
{
public:
    using ViewType = PVview;
    using ParticleType = Particle;

    TriplewiseDummyHandler(real rc, real epsilon) : ParticleFetcher(rc), epsilon_(epsilon) { }

    __D__ inline std::array<real3, 3> operator()(ParticleType p, ParticleType pA, ParticleType pB, int id, int idA, int idB) const
    {
        //this will be already calculated in driver.h
        /*
        const real3 drA  = p.r - pA.r;
        const real3 drB  = p.r - pB.r;
        const real3 drAB = pA.r - pB.r;
        const real drA2  = dot(drA, drA);
        const real drB2  = dot(drB, drB);
        const real drAB2 = dot(drAB, drAB);
        if (drA2 >= rc2_ || drB2 >= rc2_ || drAB2 >= rc2_)
            return {zero, zero, zero};
        */
        return {
            make_real3(epsilon_, 0.0_r, 0.0_r),
            make_real3(epsilon_, 0.0_r, 0.0_r),
            make_real3(epsilon_, 0.0_r, 0.0_r),
        };
    }
private:
    real epsilon_;
};

class TriplewiseDummy : public TriplewiseKernelBase
{
public:
    using HandlerType = TriplewiseDummyHandler; ///< handler type corresponding to this object
    using ParamsType  = DummyParams; ///< parameters that are used to create this object
    using ViewType    = PVview;     ///< compatible view type

    /// Constructor
    TriplewiseDummy(real rc, real epsilon) : handler_(rc, epsilon) {}

    /// Generic constructor
    TriplewiseDummy(real rc, const ParamsType& p, __UNUSED long seed=42424242) :
        TriplewiseDummy(rc, p.epsilon)
    {}

    /// get the handler that can be used on device
    const HandlerType& handler() const { return handler_; }

private:
    TriplewiseDummyHandler handler_;
};

} // namespace mirheo
