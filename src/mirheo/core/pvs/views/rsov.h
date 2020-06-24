// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "rov.h"
#include <mirheo/core/pvs/rigid_ashape_object_vector.h>

namespace mirheo
{

/** \brief A \c ROVview with additional analytic shape infos
    \tparam Shape the analytical shape that represents the object shape
 */
template <class Shape>
struct RSOVview : public ROVview
{
    /** \brief Construct a \c RSOVview
        \param [in] rsov The RigidShapedObjectVector that the view represents
        \param [in] lrov The LocalRigidObjectVector that the view represents
    */
    RSOVview(RigidShapedObjectVector<Shape> *rsov, LocalRigidObjectVector *lrov) :
        ROVview(rsov, lrov),
        shape(rsov->getShape())
    {}

    Shape shape; ///< Represents the object shape
};

/** \brief A \c RSOVview with additional rigid object info from previous time step
    \tparam Shape the analytical shape that represents the object shape
 */
template <class Shape>
struct RSOVviewWithOldMotion : public RSOVview<Shape>
{
    /** \brief Construct a \c RSOVview
        \param [in] rsov The RigidShapedObjectVector that the view represents
        \param [in] lrov The LocalRigidObjectVector that the view represents

        \rst
        .. warning::
            The rov must hold old motions channel.
        \endrst
    */
    RSOVviewWithOldMotion(RigidShapedObjectVector<Shape> *rsov, LocalRigidObjectVector *lrov) :
        RSOVview<Shape>(rsov, lrov)
    {
        old_motions = lrov->dataPerObject.getData<RigidMotion>(channel_names::oldMotions)->devPtr();
    }

    RigidMotion *old_motions {nullptr}; ///< rigid object states at previous time step
};

} // namespace mirheo
