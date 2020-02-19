// Yo ho ho ho
#define private public

#include "../timer.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/domain.h>
#include <mirheo/core/exchangers/api.h>
#include <mirheo/core/integrators/factory.h>
#include <mirheo/core/interactions/pairwise/factory.h>
#include <mirheo/core/interactions/pairwise/base_pairwise.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/particle_vector.h>

#include <gtest/gtest.h>

#include <memory>

using namespace mirheo;

const real dt = 0.0025;
const real kBT = 0.0; // to get rid of rng
const real gammadpd = 20;
const real adpd = 50;
const real powerdpd = 1.0;

const real sigma = math::sqrt(2 * gammadpd * kBT);
const real sigmaf = sigma / math::sqrt(dt);


void makeCells(real4*& pos, real4*& vel,
               real4*& posBuffer, real4*& velBuffer,
               int *cellsStartSize, int *cellsSize,
               int np, CellListInfo cinfo)
{
    for (int i = 0; i < cinfo.totcells+1; i++)
        cellsSize[i] = 0;

    for (int i = 0; i < np; i++)
        cellsSize[cinfo.getCellId(make_real3(pos[i]))]++;

    cellsStartSize[0] = 0;
    for (int i = 1; i <= cinfo.totcells; i++)
        cellsStartSize[i] = cellsSize[i-1] + cellsStartSize[i-1];

    for (int i = 0; i < np; i++)
    {
        const int cid = cinfo.getCellId(make_real3(pos[i]));
        posBuffer[cellsStartSize[cid]] = pos[i];
        velBuffer[cellsStartSize[cid]] = vel[i];
        cellsStartSize[cid]++;
    }

    for (int i = 0; i < cinfo.totcells; i++)
        cellsStartSize[i] -= cellsSize[i];

    std::swap(pos, posBuffer);
    std::swap(vel, velBuffer);
}

void integrate(real4* pos, real4 *vel, Force *accs,
               int np, real dt, DomainInfo dinfo)
{
    real3 dstart = dinfo.globalStart;
    real3 dlength = dinfo.localSize;
    
    for (int i = 0; i < np; i++)
    {
        auto& r = pos[i];
        auto& u = vel[i];
        u.x += accs[i].f.x * dt;
        u.y += accs[i].f.y * dt;
        u.z += accs[i].f.z * dt;

        r.x += u.x * dt;
        r.y += u.y * dt;
        r.z += u.z * dt;
        
        if (r.x >  dstart.x+dlength.x) r.x -= dlength.x;
        if (r.x <= dstart.x)	       r.x += dlength.x;

        if (r.y >  dstart.y+dlength.y) r.y -= dlength.y;
        if (r.y <= dstart.y)	       r.y += dlength.y;

        if (r.z >  dstart.z+dlength.z) r.z -= dlength.z;
        if (r.z <= dstart.z)	       r.z += dlength.z;
    }
}


template<typename T>
T minabs(T arg)
{
    return arg;
}

template<typename T, typename... Args>
T minabs(T arg, Args... other)
{
    const T v = minabs(other...	);
    return (std::abs(arg) < std::abs(v)) ? arg : v;
}


void forces(const real4 *pos, const real4 *vel, Force *accs,
            const int *cellsStartSize, const int *cellsSize,
            CellListInfo cinfo, DomainInfo dinfo)
{
    real3 dlength = dinfo.localSize;
    
    auto addForce = [=] (int dstId, int srcId, Force& a)
    {
        Particle pdst(pos[dstId], vel[dstId]);
        Particle psrc(pos[srcId], vel[srcId]);
        
        real _xr = pdst.r.x - psrc.r.x;
        real _yr = pdst.r.y - psrc.r.y;
        real _zr = pdst.r.z - psrc.r.z;

        _xr = minabs(_xr, _xr - dlength.x, _xr + dlength.x);
        _yr = minabs(_yr, _yr - dlength.y, _yr + dlength.y);
        _zr = minabs(_zr, _zr - dlength.z, _zr + dlength.z);

        const real rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

        if (rij2 > 1.0f) return;
        //assert(rij2 < 1);

        const real invrij = 1.0f / math::sqrt(rij2);
        const real rij = rij2 * invrij;
        const real argwr = 1.0f - rij;
        const real wr = pow(argwr, powerdpd);

        const real xr = _xr * invrij;
        const real yr = _yr * invrij;
        const real zr = _zr * invrij;

        const real rdotv =
        xr * (pdst.u.x - psrc.u.x) +
        yr * (pdst.u.y - psrc.u.y) +
        zr * (pdst.u.z - psrc.u.z);

        const real myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

        const real strength = adpd * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

        a.f.x += strength * xr;
        a.f.y += strength * yr;
        a.f.z += strength * zr;
    };

    const int3 ncells = cinfo.ncells;

#pragma omp parallel for collapse(3)
    for (int cx = 0; cx < ncells.x; cx++)
        for (int cy = 0; cy < ncells.y; cy++)
            for (int cz = 0; cz < ncells.z; cz++)
            {
                const int cid = cinfo.encode(cx, cy, cz);

                for (int dstId = cellsStartSize[cid]; dstId < cellsStartSize[cid] + cellsSize[cid]; dstId++)
                {
                    Force f (make_real4(0.f, 0.f, 0.f, 0.f));

                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                int ncx, ncy, ncz;
                                ncx = (cx+dx + ncells.x) % ncells.x;
                                ncy = (cy+dy + ncells.y) % ncells.y;
                                ncz = (cz+dz + ncells.z) % ncells.z;

                                const int srcCid = cinfo.encode(ncx, ncy, ncz);
                                if (srcCid >= cinfo.totcells || srcCid < 0) continue;

                                for (int srcId = cellsStartSize[srcCid]; srcId < cellsStartSize[srcCid] + cellsSize[srcCid]; srcId++)
                                {
                                    if (dstId != srcId)
                                        addForce(dstId, srcId, f);

                                    //printf("%d  %f %f %f\n", dstId, a.a[0], a.a[1], a.a[2]);
                                }
                            }

                    accs[dstId].f.x = f.f.x;
                    accs[dstId].f.y = f.f.y;
                    accs[dstId].f.z = f.f.z;
                }
            }
}

void execute(real3 length, int niters, double& l2, double& linf)
{
    cudaStream_t defStream;
    CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );
    
    // Initial cells
    
    real3 domainStart = -length / 2.0f;
    const real rc = 1.0f;
    const real mass = 1.0f;

    DomainInfo domainInfo;
    domainInfo.localSize = length;
    domainInfo.globalStart.x = -0.5f * length.x;
    domainInfo.globalStart.y = -0.5f * length.y;
    domainInfo.globalStart.z = -0.5f * length.z;

    MirState state(domainInfo, dt);
    
    ParticleVector pv(&state, "pv", mass);
    PrimaryCellList cells(&pv, rc, length);
    const int3 ncells = cells.ncells;

    const int ndens = 8;
    pv.local()->resize(ncells.x*ncells.y*ncells.z * ndens, defStream);
    auto& pos = pv.local()->positions();
    auto& vel = pv.local()->velocities();

    srand48(0);
    
    printf("initializing...\n");

    int c = 0;
    for (int i = 0; i < ncells.x; i++)
        for (int j = 0; j < ncells.y; j++)
            for (int k = 0; k < ncells.z; k++)
                for (int l = 0; l < ndens; l++)
                {
                    Particle p;
                    
                    p.r.x = i + drand48() + domainStart.x;
                    p.r.y = j + drand48() + domainStart.y;
                    p.r.z = k + drand48() + domainStart.z;                    

                    p.u.x = 0*(drand48() - 0.5);
                    p.u.y = 0*(drand48() - 0.5);
                    p.u.z = 0*(drand48() - 0.5);
                    p.setId(c);

                    pos[c] = p.r2Real4();
                    vel[c] = p.u2Real4();
                    c++;
                }


    pos.uploadToDevice(defStream);
    vel.uploadToDevice(defStream);
    pv.local()->forces().clear(defStream);

    HostBuffer<real4> positions(pv.local()->size()), velocities(pv.local()->size());
    std::copy(pos.begin(), pos.end(), positions .begin());
    std::copy(vel.begin(), vel.end(), velocities.begin());

    auto haloExchanger = std::make_unique<ParticleHaloExchanger>();
    haloExchanger->attach(&pv, &cells, {});
    SingleNodeEngine haloEngine(std::move(haloExchanger));

    auto redistributor = std::make_unique<ParticleRedistributor>();
    redistributor->attach(&pv, &cells);
    SingleNodeEngine redistEngine(std::move(redistributor));

    const DPDParams dpdParams{adpd, gammadpd, kBT, powerdpd};
    auto dpd = createInteractionPairwise(&state, "dpd", rc, dpdParams, StressNoneParams{});

    auto integrator = IntegratorFactory::createVV(&state, "vv");
    
    CUDA_Check( cudaStreamSynchronize(defStream) );

    printf("GPU execution\n");

    Timer tm;
    tm.start();

    for (int i = 0; i < niters; i++)
    {
        state.currentStep = i;
        state.currentTime = i * dt;
        
        pv.local()->forces().clear(defStream);
        cells.build(defStream);

        haloEngine.init(defStream);
        
        dpd->setPrerequisites(&pv, &pv, &cells, &cells);
        dpd->local(&pv, &pv, &cells, &cells, defStream);

        haloEngine.finalize(defStream);

        dpd->halo(&pv, &pv, &cells, &cells, defStream);

        integrator->setPrerequisites(&pv);
        integrator->execute(&pv, defStream);
        
        CUDA_Check( cudaStreamSynchronize(defStream) );

        redistEngine.init(defStream);
        redistEngine.finalize(defStream);

        CUDA_Check( cudaStreamSynchronize(defStream) );
    }

    double elapsed = tm.elapsed() * 1e-9;

    printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);

    cells.build(defStream);

    int np = positions.size();
    int totcells = cells.totcells;

    HostBuffer<real4> posBuffer(np), velBuffer(np);
    HostBuffer<Force> accs(np);
    HostBuffer<int>   cellsStartSize(totcells+1), cellsSize(totcells+1);
    
    printf("CPU execution\n");
    
    for (int i = 0; i < niters; i++)
    {
        printf("%d...", i);
        fflush(stdout);

        makeCells(positions.hostPtr_, velocities.hostPtr_,
                  posBuffer.hostPtr_, velBuffer.hostPtr_,
                  cellsStartSize.data(), cellsSize.data(), np, cells.cellInfo());

        forces(positions.data(), velocities.data(),
               accs.data(), cellsStartSize.data(), cellsSize.data(), cells.cellInfo(), domainInfo);

        integrate(positions.data(), velocities.data(), accs.data(), np, dt, domainInfo);
    }

    printf("\nDone, checking\n");
    printf("NP:  %d,  ref  %d\n", pv.local()->size(), np);


    pos.downloadFromDevice(defStream, ContainersSynch::Asynch);
    vel.downloadFromDevice(defStream, ContainersSynch::Synch);

    std::vector<int> gpuid(np), cpuid(np);
    for (int i = 0; i < np; i++)
    {
        Particle pg(pos[i], vel[i]);
        Particle pc(positions[i], velocities[i]);
        
        gpuid[pg.getId()] = i;
        cpuid[pc.getId()] = i;
    }


    l2 = 0;
    linf = -1;

    for (int i = 0; i < np; i++)
    {
        Particle cpuP(positions[cpuid[i]], velocities[cpuid[i]]);
        Particle gpuP(pos[gpuid[i]], vel[gpuid[i]]);

        double perr = -1;


        double3 err = {
            math::abs(cpuP.r.x - gpuP.r.x) + math::abs(cpuP.u.x - gpuP.u.x),
            math::abs(cpuP.r.y - gpuP.r.y) + math::abs(cpuP.u.y - gpuP.u.y),
            math::abs(cpuP.r.z - gpuP.r.z) + math::abs(cpuP.u.z - gpuP.u.z)};
            
        linf = max(linf, max(err.x, max(err.y, err.z)));
        perr = max(perr, max(err.x, max(err.y, err.z)));
        l2 += err.x * err.x + err.y * err.y + err.z * err.z;

        if (perr > 0.01)
        {
            printf("id %8d diff %8e  [%12f %12f %12f  %8d] [%12f %12f %12f]\n"
                   "                           ref [%12f %12f %12f  %8d] [%12f %12f %12f] \n\n", i, perr,
                   gpuP.r.x, gpuP.r.y, gpuP.r.z, gpuP.i1,
                   gpuP.u.x, gpuP.u.y, gpuP.u.z,
                   cpuP.r.x, cpuP.r.y, cpuP.r.z, cpuP.i1,
                   cpuP.u.x, cpuP.u.y, cpuP.u.z);
        }
    }

    l2 = math::sqrt(l2 / pv.local()->size());
    printf("L2   norm: %f\n", l2);
    printf("Linf norm: %f\n", linf);
}

TEST (ONE_RANK, small)
{
    double l2, linf, tol;
    int niters = 50;
    real3 length{8, 8, 8};
    tol = 0.001;
    
    execute(length, niters, l2, linf);

    ASSERT_LE(l2,   tol);
    ASSERT_LE(linf, tol);
}

TEST (ONE_RANK, big)
{
    double l2, linf, tol;
    int niters = 3;
    real3 length{32, 32, 32};
    tol = 0.00002;
    
    execute(length, niters, l2, linf);

    ASSERT_LE(l2,   tol);
    ASSERT_LE(linf, tol);
}

int main(int argc, char ** argv)
{
    int provided, required = MPI_THREAD_FUNNELED;
    MPI_Init_thread(&argc, &argv, required, &provided);

    if (provided < required) {
        printf("ERROR: The MPI library does not have required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    logger.init(MPI_COMM_WORLD, "onerank.log", 9);

    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();
    
    MPI_Finalize();
    return retval;
}
