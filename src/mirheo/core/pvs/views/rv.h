// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "ov.h"

namespace mirheo
{

class RodVector;
class LocalRodVector;

/// A \c OVview with additional rod object infos
struct RVview : public OVview
{
    /** \brief Construct a \c RVview
        \param [in] rv The RodVector that the view represents
        \param [in] lrv The LocalRodVector that the view represents
    */
    RVview(RodVector *rv, LocalRodVector *lrv);

    int   nSegments {0}; ///< number of segments per rod
    int   *states   {nullptr}; ///< polymorphic states per bisegment
    real *energies {nullptr}; ///< energies per bisegment
};

/// A \c RVview with additional particles from previous time step
struct RVviewWithOldParticles : public RVview
{
    /** \brief Construct a \c RVview
        \param [in] rv The RodVector that the view represents
        \param [in] lrv The LocalRodVector that the view represents
    */
    RVviewWithOldParticles(RodVector *rv, LocalRodVector *lrv);

    real4 *oldPositions {nullptr}; ///< positions o the particles at previous time step
};

} // namespace mirheo
