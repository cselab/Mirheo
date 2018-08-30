/*
 * bounce.cu
 *
 *  Created on: Jul 20, 2017
 *      Author: alexeedm
 */

#include "from_ellipsoid.h"

#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/rigid_ellipsoid_object_vector.h>
#include <core/pvs/views/reov.h>

#include <core/rigid_kernels/bounce.h>
#include <core/rigid_kernels/integration.h>


/**
 * Create the bouncer
 * @param name unique bouncer name
 */
BounceFromRigidEllipsoid::BounceFromRigidEllipsoid(std::string name) : Bouncer(name)
{    }

/**
 * @param ov will need an 'old_motions' channel with the rigid motion
 * from the previous timestep, to be used in bounceEllipsoid()
 */
void BounceFromRigidEllipsoid::setup(ObjectVector* ov)
{
    this->ov = ov;

    ov->requireDataPerObject<RigidMotion> ("old_motions", true, sizeof(RigidReal));
}

/**
 * Calls ObjectVector::findExtentAndCOM and then calls
 * bounceEllipsoid() function
 */
void BounceFromRigidEllipsoid::exec(ParticleVector* pv, CellList* cl, float dt, bool local, cudaStream_t stream)
{
    auto reov = dynamic_cast<RigidEllipsoidObjectVector*>(ov);
    if (reov == nullptr)
        die("Analytic ellispoid bounce only works with Rigid Ellipsoids");

    debug("Bouncing %d '%s' particles from %d '%s': objects (%s)",
            pv->local()->size(), pv->name.c_str(),
            local ? reov->local()->nObjects : reov->halo()->nObjects, reov->name.c_str(),
            local ? "local objs" : "halo objs");

    ov->findExtentAndCOM(stream, local ? ParticleVectorType::Local : ParticleVectorType::Halo);

    REOVviewWithOldMotion ovView(reov, local ? reov->local() : reov->halo());
    PVviewWithOldParticles pvView(pv, pv->local());

    int nthreads = 256;
    if (!local)
    {
        SAFE_KERNEL_LAUNCH(
                clearRigidForces,
                getNblocks(ovView.nObjects, nthreads), nthreads, 0, stream,
                ovView );
    }

    SAFE_KERNEL_LAUNCH(
            bounceEllipsoid,
            ovView.nObjects, nthreads, 2*nthreads*sizeof(int), stream,
            ovView, pvView, cl->cellInfo(), dt );
}



