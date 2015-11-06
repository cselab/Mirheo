/*
 *  main.cu
 *  ctc PANDA
 *
 *  Created by Dmitry Alexeev on Oct 20, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

/*
 *  main.cu
 *  Part of uDeviceX/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2014-11-14.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>
#include <csignal>
#include <mpi.h>
#include <errno.h>
#include <vector>

#include <argument-parser.h>
#include <common.h>
#include <containers.h>
#include <contact.h>
#include <dpd-rng.h>
#include <solute-exchange.h>
 enum
    {
	XCELLS = XSIZE_SUBDOMAIN,
	YCELLS = YSIZE_SUBDOMAIN,
	ZCELLS = ZSIZE_SUBDOMAIN,
	XOFFSET = XCELLS / 2,
	YOFFSET = YCELLS / 2,
	ZOFFSET = ZCELLS / 2
    };
using namespace std;

float tend, couette;
bool walls, pushtheflow, doublepoiseuille, rbcs, ctcs, xyz_dumps, hdf5field_dumps, hdf5part_dumps, is_mps_enabled, adjust_message_sizes, contactforces, stress;
int steps_per_report, steps_per_dump, wall_creation_stepid, nvtxstart, nvtxstop;

LocalComm localcomm;

static const float ljsigma = 0.5;
static const float ljsigma2 = ljsigma * ljsigma;

template<int s>
inline  float _viscosity_function(float x)
{
    return sqrtf(viscosity_function<s - 1>(x));
}

template<> inline  float _viscosity_function<1>(float x) { return sqrtf(x); }
template<> inline  float _viscosity_function<0>(float x){ return x; }

int main(int argc, char ** argv)
{
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceReset());

    {
        is_mps_enabled = false;

        const char * mps_variables[] = {
                "CRAY_CUDA_MPS",
                "CUDA_MPS",
                "CRAY_CUDA_PROXY",
                "CUDA_PROXY"
        };

        for(int i = 0; i < 4; ++i)
            is_mps_enabled |= getenv(mps_variables[i])!= NULL && atoi(getenv(mps_variables[i])) != 0;
    }

    int nranks, rank;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    MPI_Comm activecomm = MPI_COMM_WORLD;

    bool reordering = true;
    const char * env_reorder = getenv("MPICH_RANK_REORDER_METHOD");

    MPI_Comm cartcomm;
    int periods[] = {1, 1, 1};
    int ranks[] = {1, 1, 1};


    MPI_CHECK( MPI_Cart_create(activecomm, 3, ranks, periods, (int)reordering, &cartcomm) );
    activecomm = cartcomm;

    {
        MPI_CHECK(MPI_Barrier(activecomm));
        localcomm.initialize(activecomm);

        MPI_CHECK(MPI_Barrier(activecomm));

        // test here
	const size_t myseed = 0x563d00cf;//time(NULL);
        srand48(myseed);
	printf("myseed: 0x%x\n", myseed);

        int n = 25e3;
        vector<Particle> ic(n);
        vector<Acceleration> acc(n);
        for (int i=0; i<n; i++)
            for (int d=0; d<3; d++)
                acc[i].a[d] = 0;
        vector<Acceleration> gpuacc(n);

	Logistic::KISS local_trunk = Logistic::KISS(7119 - rank, 187 + rank, 18278, 15674);

     	const double center[3] = { -XSIZE_SUBDOMAIN/2, -YSIZE_SUBDOMAIN/2, -ZSIZE_SUBDOMAIN/2} ;//YSIZE_SUBDOMAIN/2 -};
	const double halfwidth[3] = {XSIZE_SUBDOMAIN/10., YSIZE_SUBDOMAIN/10., ZSIZE_SUBDOMAIN / 10.};

	for(int i = 0; i < n; ++i)
	{
	    ic[i].x[0] = center[0] + halfwidth[0] * 2 * (drand48() - 0.5);
	    ic[i].x[1] = center[1] + halfwidth[1] * 2 * (drand48() - 0.5);
	    ic[i].x[2] = center[2] + halfwidth[2] * 2 * (drand48() - 0.5);
	    ic[i].u[0] = 0.5 - drand48();
	    ic[i].u[1] = 0.5 - drand48();
	    ic[i].u[2] = 0.5 - drand48();
	}

	if (false)//if (true)
	{
	    ic.resize(2);
	    acc.resize(2);
	    gpuacc.resize(2);
	    n = 2;
	    //ic[0].x[0] = 24.413; ic[0].x[1] = +14.924; ic[0].x[2] = +7.326;
	    //ic[1].x[0] = +23.895; ic[1].x[1] = +14.887; ic[1].x[2] = +7.455 ;

	    ic[0].x[0] = +23.670 ; ic[0].x[1] =+23.494; ic[0].x[2] =-18.980;
	    ic[1].x[0] = -23.851 ; ic[1].x[1] =+23.696; ic[1].x[2] =-18.790;
	}

        float seed = local_trunk.get_float();

#pragma omp parallel for
        for (int i=0; i<n; i++)
            for (int j=0; j<n; j++)
                if (i != j)
                {
                    double _xr = ic[i].x[0] - ic[j].x[0];
                    double _yr = ic[i].x[1] - ic[j].x[1];
                    double _zr = ic[i].x[2] - ic[j].x[2];

		    _xr -= XSIZE_SUBDOMAIN * floor(0.5 + _xr / XSIZE_SUBDOMAIN);
		    _yr -= YSIZE_SUBDOMAIN * floor(0.5 + _yr / YSIZE_SUBDOMAIN);
		    _zr -= ZSIZE_SUBDOMAIN * floor(0.5 + _zr / ZSIZE_SUBDOMAIN);

		    const double rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
                    const double invrij = rsqrtf(rij2);
                    const double rij = rij2 * invrij;

                    if (rij2 >= 1)
                        continue;

                    const double invr2 = invrij * invrij;
                    const double t2 = ljsigma2 * invr2;
                    const double t4 = t2 * t2;
                    const double t6 = t4 * t2;
                    const double lj = min(1e4f, max(0.f, 24.f * invrij * t6 * (2.f * t6 - 1.f)));

                    const double wr = _viscosity_function<0>(1.f - rij);

                    const double xr = _xr * invrij;
                    const double yr = _yr * invrij;
                    const double zr = _zr * invrij;

                    const double strength = lj;

                    const double xinteraction = strength * xr;
                    const double yinteraction = strength * yr;
                    const double zinteraction = strength * zr;

                    acc[i].a[0] += xinteraction;
                    acc[i].a[1] += yinteraction;
                    acc[i].a[2] += zinteraction;
		}

        ParticleArray p;
        p.resize(n);

        CUDA_CHECK( cudaMemcpy(p.xyzuvw.data, &ic[0], n * sizeof(Particle), cudaMemcpyHostToDevice) );
        CUDA_CHECK( cudaMemset(p.axayaz.data, 0, n * sizeof(Acceleration)) );
        std::vector<ParticlesWrap> wsolutes;
        wsolutes.push_back(ParticlesWrap(p.xyzuvw.data, n, p.axayaz.data));

        ComputeContact contact(cartcomm);
        SoluteExchange solutex(cartcomm);

        solutex.attach_halocomputation(contact);
        contact.attach_bulk(wsolutes);

        solutex.bind_solutes(wsolutes);
        solutex.pack_p(0);
        solutex.post_p(0, 0);
        solutex.recv_p(0);
        solutex.halo(0, 0);
        solutex.post_a();
        solutex.recv_a(0);

        CUDA_CHECK( cudaMemcpy(&gpuacc[0], p.axayaz.data, n * sizeof(Acceleration), cudaMemcpyDeviceToHost) );

	{
	    double fx = 0, fy = 0, fz = 0;
	    double hfx = 0, hfy = 0, hfz = 0;

	    for (int i=0; i<n; i++)
	    {
		fx += gpuacc[i].a[0];
		fy += gpuacc[i].a[1];
		fz += gpuacc[i].a[2];

		hfx += acc[i].a[0];
		hfy += acc[i].a[1];
		hfz += acc[i].a[2];
	    }

	    printf("CPU F [%8f %8f %8f], GPU F [%8f %8f %8f]\n", hfx, hfy, hfz, fx, fy, fz);
	    printf("%d particles.\n", n);
	}

	{
	    const float * ref = &acc[0].a[0];
	    const float * res = &gpuacc[0].a[0];
	    const double tol = 5e-3;

	    double linf = 0, l1 = 0, linf_rel = 0, l1_rel = 0;
	    const int nentries = 3 * n;
	    bool failed = false;
	    for(int i = 0; i < nentries; ++i)
	    {
		assert(!std::isnan(ref[i]));
		assert(!std::isnan(res[i]));

		const double err = ref[i] - res[i];
		const double maxval = std::max(fabs(res[i]), fabs(ref[i]));
		const double relerr = err/std::max(1e-6, maxval);

		failed |= fabs(relerr) >= tol && fabs(err) >= tol;

		if (failed)
		    printf("p %d c %d: %e ref: %e -> %e %e\n", i / 3, i % 3, res[i], ref[i], err, relerr);

		if (i % 3 == 2 && failed)
		{
		    const int pid = i/3;

		    const bool inside =
			ic[i].x[0] >= -XOFFSET && ic[pid].x[0] < XOFFSET &&
			ic[i].x[1] >= -YOFFSET && ic[pid].x[1] < YOFFSET &&
			ic[i].x[2] >= -ZOFFSET && ic[pid].x[2] < ZOFFSET ;

		    printf("%d:  CPU [%+.3f %+.3f %+.3f]  GPU [%+.3f %+.3f %+.3f] -> p %+.3f %+.3f %+.3f -> inside: %d\n",
			   i, acc[pid].a[0], acc[pid].a[1], acc[pid].a[2],
			   gpuacc[pid].a[0], gpuacc[pid].a[1], gpuacc[pid].a[2],
			   ic[pid].x[0], ic[pid].x[1], ic[pid].x[2], inside);

		    failed = false;
		}

		assert(fabs(relerr) < tol || fabs(err) < tol);

		l1 += fabs(err);
		l1_rel += fabs(relerr);

		linf = std::max(linf, fabs(err));
		linf_rel = std::max(linf_rel, fabs(relerr));
	    }

	    printf("l-infinity errors: %.03e (absolute) %.03e (relative)\n", linf, linf_rel);
	    printf("       l-1 errors: %.03e (absolute) %.03e (relative)\n", l1, l1_rel);
	}
    }

    if (activecomm != cartcomm)
        MPI_CHECK(MPI_Comm_free(&activecomm));

    MPI_CHECK(MPI_Comm_free(&cartcomm));

    MPI_CHECK(MPI_Finalize());

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}