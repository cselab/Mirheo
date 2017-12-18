#include "rbc.h"

#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/celllist.h>
#include <core/pvs/rbc_vector.h>

#include <core/rbc_kernels/interactions.h>

#include <cmath>

/**
 * Provide mapping from the model parameters to the parameters
 * used on GPU to compute the forces.
 *
 * Scaling accorging to the Mesh properties is performed, and some
 * thing are pre-computed.
 *
 * @param p model parameters
 * @param m RBC membrane mesh
 * @return parameters to be passed to GPU kernels
 */
static GPU_RBCparameters setParams(RBCParameters p, const Mesh& m)
{
	GPU_RBCparameters devP;

	devP.gammaC = p.gammaC;
	devP.gammaT = p.gammaT;

	devP.area0 = p.totArea0 / m.ntriangles;
	devP.totArea0 = p.totArea0;
	devP.totVolume0 = p.totVolume0;

	devP.mpow = p.mpow;
	auto l0 = sqrt(devP.area0 * 4.0 / sqrt(3.0));
	devP.lmax = l0 / p.x0;
	devP.kbToverp = p.kbT / p.p;

	devP.cost0kb = cos(p.theta / 180.0 * M_PI) * p.kb;
	devP.sint0kb = sin(p.theta / 180.0 * M_PI) * p.kb;

	devP.ka0 = p.ka / p.totArea0;
	devP.kv0 = p.kv / (6.0*p.totVolume0);
	devP.kd0 = p.kd;

	devP.kp0 = ( p.kbT * p.x0 * (4*p.x0*p.x0 - 9*p.x0 + 6) * l0*l0) / (4*p.p * sqr(p.x0 - 1) );

	return devP;
}

/**
 * Require that \p pv1 and \p pv2 are the same and are instances
 * of RBCvector
 */
void InteractionRBCMembrane::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
	if (pv1 != pv2)
		die("Internal RBC forces can't be computed between two different particle vectors");

	auto ov = dynamic_cast<RBCvector*>(pv1);
	if (ov == nullptr)
		die("Internal RBC forces can only be computed with RBCs");
}

/**
 * Compute the membrane forces.
 *
 * First call the computeAreaAndVolume() kernel to compute area
 * and volume of each cell, then use these data to calculate the
 * forces themselves by calling computeMembraneForces() kernel
 */
void InteractionRBCMembrane::regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	auto ov = dynamic_cast<RBCvector*>(pv1);

	if (ov->objSize != ov->mesh.nvertices)
		die("Object size of '%s' (%d) and number of vertices (%d) mismatch",
				ov->name.c_str(), ov->objSize, ov->mesh.nvertices);

	debug("Computing internal membrane forces for %d cells of '%s'",
		ov->local()->nObjects, ov->name.c_str());

	OVviewWithAreaVolume view(ov, ov->local());
	MeshView mesh(ov->mesh);
	ov->local()->extraPerObject.getData<float2>("area_volumes")->clearDevice(stream);

	const int nthreads = 128;
	SAFE_KERNEL_LAUNCH(
			computeAreaAndVolume,
			view.nObjects, nthreads, 0, stream,
			view, mesh );


	const int blocks = getNblocks(view.size, nthreads);
	SAFE_KERNEL_LAUNCH(
			computeMembraneForces<Mesh::maxDegree>,
			blocks, nthreads, 0, stream,
			view, mesh, setParams(parameters, ov->mesh) );
}

void InteractionRBCMembrane::halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	debug("Not computing internal RBC forces between local and halo RBCs of '%s'", pv1->name.c_str());
}



