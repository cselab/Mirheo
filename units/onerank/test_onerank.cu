// Yo ho ho ho
#define private public

#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/domain.h>
#include <core/mpi/api.h>
#include <core/logger.h>
#include <core/integrators/factory.h>

#include "../timer.h"
#include <unistd.h>

Logger logger;

void makeCells(Particle*& __restrict__ coos, Particle*& __restrict__ buffer, int* __restrict__ cellsStartSize, int* __restrict__ cellsSize,
               int np, CellListInfo cinfo)
{
    for (int i = 0; i < cinfo.totcells+1; i++)
        cellsSize[i] = 0;

    for (int i = 0; i < np; i++)
        cellsSize[cinfo.getCellId(coos[i].r)]++;

    cellsStartSize[0] = 0;
    for (int i = 1; i <= cinfo.totcells; i++)
        cellsStartSize[i] = cellsSize[i-1] + cellsStartSize[i-1];

    for (int i = 0; i < np; i++)
    {
        const int cid = cinfo.getCellId(coos[i].r);
        buffer[cellsStartSize[cid]] = coos[i];
        cellsStartSize[cid]++;
    }

    for (int i = 0; i < cinfo.totcells; i++)
        cellsStartSize[i] -= cellsSize[i];

    std::swap(coos, buffer);
}

void integrate(Particle* __restrict__ coos, Force* __restrict__ accs, int np, float dt,
               CellListInfo cinfo, DomainInfo dinfo)
{
    for (int i = 0; i < np; i++)
    {            
        coos[i].u.x += accs[i].f.x * dt;
        coos[i].u.y += accs[i].f.y * dt;
        coos[i].u.z += accs[i].f.z * dt;

        coos[i].r.x += coos[i].u.x * dt;
        coos[i].r.y += coos[i].u.y * dt;
        coos[i].r.z += coos[i].u.z * dt;

        if (coos[i].r.x >  cinfo.domainStart.x+cinfo.length.x) coos[i].r.x -= cinfo.length.x;
        if (coos[i].r.x <= cinfo.domainStart.x)				coos[i].r.x += cinfo.length.x;

        if (coos[i].r.y >  cinfo.domainStart.y+cinfo.length.y) coos[i].r.y -= cinfo.length.y;
        if (coos[i].r.y <= cinfo.domainStart.y)				coos[i].r.y += cinfo.length.y;

        if (coos[i].r.z >  cinfo.domainStart.z+cinfo.length.z) coos[i].r.z -= cinfo.length.z;
        if (coos[i].r.z <= cinfo.domainStart.z)				coos[i].r.z += cinfo.length.z;
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


void forces(const Particle* __restrict__ coos, Force* __restrict__ accs, const int* __restrict__ cellsStartSize, const int* __restrict__ cellsSize,
            CellListInfo cinfo)
{

    const float dt = 0.0025;
    const float kBT = 1.0;
    const float gammadpd = 20;
    const float sigma = sqrt(2 * gammadpd * kBT);
    const float sigmaf = sigma / sqrt(dt);
    const float aij = 50;

    auto addForce = [=] (int dstId, int srcId, Force& a)
    {
        float _xr = coos[dstId].r.x - coos[srcId].r.x;
        float _yr = coos[dstId].r.y - coos[srcId].r.y;
        float _zr = coos[dstId].r.z - coos[srcId].r.z;

        _xr = minabs(_xr, _xr - cinfo.length.x, _xr + cinfo.length.x);
        _yr = minabs(_yr, _yr - cinfo.length.y, _yr + cinfo.length.y);
        _zr = minabs(_zr, _zr - cinfo.length.z, _zr + cinfo.length.z);

        const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;

        if (rij2 > 1.0f) return;
        //assert(rij2 < 1);

        const float invrij = 1.0f / sqrt(rij2);
        const float rij = rij2 * invrij;
        const float argwr = 1.0f - rij;
        const float wr = argwr;

        const float xr = _xr * invrij;
        const float yr = _yr * invrij;
        const float zr = _zr * invrij;

        const float rdotv =
        xr * (coos[dstId].u.x - coos[srcId].u.x) +
        yr * (coos[dstId].u.y - coos[srcId].u.y) +
        zr * (coos[dstId].u.z - coos[srcId].u.z);

        const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

        const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

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
                    Force f (make_float4(0.f, 0.f, 0.f, 0.f));

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

int main(int argc, char ** argv)
{
    // Init

    int nranks, rank;
    int ranks[] = {1, 1, 1};
    int periods[] = {1, 1, 1};
    MPI_Comm cartComm;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE)
    {
        printf("ERROR: The MPI library does not have full thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    logger.init(MPI_COMM_WORLD, "onerank.log", 9);
    IniParser config("tests.cfg");

    MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

    // Initial cells

    float3 length{64, 64, 64};
    float3 domainStart = -length / 2.0f;
    const float rc = 1.0f;
    ParticleVector dpds(domainStart, length);
    CellList cells(&dpds, rc, domainStart, length);
    const int3 ncells = cells.ncells;

    const int ndens = 8;
    dpds.resize(ncells.x*ncells.y*ncells.z * ndens);

    srand48(0);

    printf("initializing...\n");

    int c = 0;
    for (int i=0; i<ncells.x; i++)
        for (int j=0; j<ncells.y; j++)
            for (int k=0; k<ncells.z; k++)
                for (int p=0; p<ndens; p++)
                {
                    dpds.local()->coosvels[c].r.x = i + drand48() + domainStart.x;
                    dpds.local()->coosvels[c].r.y = j + drand48() + domainStart.y;
                    dpds.local()->coosvels[c].r.z = k + drand48() + domainStart.z;
                    dpds.local()->coosvels[c].i1 = c;

                    dpds.local()->coosvels[c].u.x = 0*(drand48() - 0.5);
                    dpds.local()->coosvels[c].u.y = 0*(drand48() - 0.5);
                    dpds.local()->coosvels[c].u.z = 0*(drand48() - 0.5);
                    c++;
                }


    dpds.resize(c);
    dpds.local()->coosvels.synchronize(synchronizeDevice);
    dpd.local()->forces.clear();

    HostBuffer<Particle> particles(dpds.local()->size());
    for (int i=0; i<dpds.local()->size(); i++)
        particles[i] = dpds.local()->coosvels[i];

    cudaStream_t defStream;
    CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

    HaloExchanger halo(cartComm);
    halo->attach(&dpds, &cells, ndens);
    Redistributor redist(cartComm, config);
    redist.attach(&dpds, &cells, ndens);

    CUDA_Check( cudaStreamSynchronize(defStream) );

    const float dt = 0.005;
    const int niters = 200;

    printf("GPU execution\n");

    Timer tm;
    tm.start();

    for (int i=0; i<niters; i++)
    {
        dpd.local()->forces.clear(defStream);
        cells.build(defStream);
        computeInternalDPD(dpds, cells, defStream);

        halo->init();
        halo->finalize();

        computeHaloDPD(dpds, cells, defStream);

        integrateNoFlow(dpds, dt, 1.0f, defStream);
        CUDA_Check( cudaStreamSynchronize(defStream) );

        redist.redistribute();

        CUDA_Check( cudaStreamSynchronize(defStream) );
    }

    double elapsed = tm.elapsed() * 1e-9;

    printf("Finished in %f s, 1 step took %f ms\n", elapsed, elapsed / niters * 1000.0);


    if (argc < 2) return 0;

    cells.build(defStream);

    int np = particles.size;
    int totcells = cells.totcells;

    HostBuffer<Particle> buffer(np);
    HostBuffer<Force> accs(np);
    HostBuffer<int>   cellsStartSize(totcells+1), cellsSize(totcells+1);

    printf("CPU execution\n");

    for (int i=0; i<niters; i++)
    {
        printf("%d...", i);
        fflush(stdout);
        makeCells(particles.hostdata, buffer.hostdata, cellsStartSize.hostdata, cellsSize.hostdata, np, cells.cellInfo());
        forces(particles.hostdata, accs.hostdata, cellsStartSize.hostdata, cellsSize.hostdata, cells.cellInfo());
        integrate(particles.hostdata, accs.hostdata, np, dt, cells.cellInfo());
    }

    printf("\nDone, checking\n");
    printf("NP:  %d,  ref  %d\n", dpds.local()->size(), np);


    dpds.local()->coosvels.synchronize(synchronizeHost);

    std::vector<int> gpuid(np), cpuid(np);
    for (int i=0; i<np; i++)
    {
        gpuid[dpds.local()->coosvels[i].i1] = i;
        cpuid[particles[i].i1] = i;
    }


    double l2 = 0, linf = -1;

    for (int i=0; i<np; i++)
    {
        Particle cpuP = particles[cpuid[i]];
        Particle gpuP = dpds.local()->coosvels[gpuid[i]];

        double perr = -1;
        for (int c=0; c<3; c++)
        {
            const double err = fabs(cpuP.x[c] - gpuP.x[c]) + fabs(cpuP.u[c] - gpuP.u[c]);
            linf = max(linf, err);
            perr = max(perr, err);
            l2 += err * err;
        }

        if (argc > 2 && perr > 0.01)
        {
            printf("id %8d diff %8e  [%12f %12f %12f  %8d] [%12f %12f %12f]\n"
                   "                           ref [%12f %12f %12f  %8d] [%12f %12f %12f] \n\n", i, perr,
                   gpuP.r.x, gpuP.r.y, gpuP.r.z, gpuP.i1,
                   gpuP.u.x, gpuP.u.y, gpuP.u.z,
                   cpuP.r.x, cpuP.r.y, cpuP.r.z, cpuP.i1,
                   cpuP.u.x, cpuP.u.y, cpuP.u.z);
        }
    }

    l2 = sqrt(l2 / dpds.local()->size());
    printf("L2   norm: %f\n", l2);
    printf("Linf norm: %f\n", linf);

    return 0;
}
