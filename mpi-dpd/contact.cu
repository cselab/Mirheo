/*
 * contact.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-12-02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <../dpd-rng.h>

#include "contact.h"

namespace KernelsContact
{
    struct Params { float aij, gamma, sigmaf, rc2; };

    __constant__ Params params;
}

ComputeContact::ComputeContact(MPI_Comm comm)
{
    int myrank;
    MPI_CHECK( MPI_Comm_rank(comm, &myrank));

    local_trunk = Logistic::KISS(1908 - myrank, 1409 + myrank, 290, 12968);

    //CUDA_CHECK(cudaEventCreateWithFlags(&evuploaded, cudaEventDisableTiming));

    KernelsContact::Params params = {12.5 , gammadpd, sigmaf, 1};

    CUDA_CHECK(cudaMemcpyToSymbol(KernelsContact::params, &params, sizeof(params)));

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeContact::bulk(std::vector<ParticlesWrap> wsolutes, cudaStream_t stream)
{
    NVTX_RANGE("Contact/bulk", NVTX_C6);

    if (wsolutes.size() == 0)
	return;

    CUDA_CHECK(cudaPeekAtLastError());
}

void ComputeContact::halo(ParticlesWrap halos[26], cudaStream_t stream)
{
    NVTX_RANGE("Contact/halo", NVTX_C7);

    CUDA_CHECK(cudaPeekAtLastError());
}
