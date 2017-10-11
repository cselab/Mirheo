#include "rbc.h"

#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/pvs/rbc_vector.h>

#include <core/rbc_kernels/interactions.h>

InteractionRBCMembrane::InteractionRBCMembrane(pugi::xml_node node)
{
	name = node.attribute("name").as_string("");

	// TODO: parameter setup
}

void InteractionRBCMembrane::_compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	if (pv1 != pv2)
		die("Internal RBC forces can't be computed between two different particle vectors");

	auto rbcv = dynamic_cast<RBCvector*>(pv1);
	if (rbcv == nullptr)
		die("Internal RBC forces can only be computed with RBC object vector");

	int nthreads = 128;
	int nRbcs  = rbcv->local()->nObjects;
	int nVerts = rbcv->mesh.nvertices;


	dim3 avThreads(256, 1);
	dim3 avBlocks( 1, nRbcs );
	computeAreaAndVolume <<< avBlocks, avThreads, 0, stream >>> (
			(float4*)rbcv->local()->coosvels.devPtr(), rbcv->mesh, nRbcs,
			rbcv->local()->areas.devPtr(), rbcv->local()->volumes.devPtr());

	int blocks = getNblocks(nRbcs*nVerts*rbcv->mesh.maxDegree, nthreads);

	computeMembraneForces <<<blocks, nthreads, 0, stream>>> (
			(float4*)rbcv->local()->coosvels.devPtr(), rbcv->mesh, nRbcs,
			rbcv->local()->areas.devPtr(), rbcv->local()->volumes.devPtr(),
			(float4*)rbcv->local()->forces.devPtr());
}



