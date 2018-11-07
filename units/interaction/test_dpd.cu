#define protected public
#define private public

#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/containers.h>
#include <core/interactions/pairwise.h>
#include <core/interactions/pairwise_interactions/norandom_dpd.h>
#include <core/initial_conditions/uniform_ic.h>

#include <unistd.h>

Logger logger;

void makeCells(const Particle* __restrict__ input, Particle* __restrict__ output, int* __restrict__ cellsStartSize, int* __restrict__ cellsSize,
               int* __restrict__ order, int np, CellListInfo cinfo)
{
    for (int i=0; i<cinfo.totcells+1; i++)
        cellsSize[i] = 0;

    for (int i=0; i<np; i++)
        cellsSize[cinfo.getCellId(input[i].r)]++;

    cellsStartSize[0] = 0;
    for (int i=1; i<=cinfo.totcells; i++)
        cellsStartSize[i] = cellsSize[i-1] + cellsStartSize[i-1];

    for (int i=0; i<np; i++)
    {
        const int cid = cinfo.getCellId(input[i].r);
        output[cellsStartSize[cid]] = input[i];
        order[cellsStartSize[cid]] = i;

        cellsStartSize[cid]++;
    }

    for (int i=0; i<cinfo.totcells; i++)
        cellsStartSize[i] -= cellsSize[i];
}


int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "dpd.log", 9);

    float3 length{80, 64, 29};
    DomainInfo domain{length, {0,0,0}, length};

    const float rc = 1.0f;
    ParticleVector dpds1("dpd1", 1.0f);
    ParticleVector dpds2("dpd2", 1.0f);

    UniformIC ic1(4.5);
    UniformIC ic2(3.5);
    ic1.exec(MPI_COMM_WORLD, &dpds1, domain, 0);
    ic2.exec(MPI_COMM_WORLD, &dpds2, domain, 0);

    cudaDeviceSynchronize();

    const int np = dpds1.local()->size() + dpds2.local()->size();

    fprintf(stderr, " First part %d, second %d, total %d\n", dpds1.local()->size(), dpds2.local()->size(), np);

    CellList* cells1 = new PrimaryCellList(&dpds1, rc, length);
    CellList* cells2 = new PrimaryCellList(&dpds2, rc, length);
    cells1->build(0);
    cells2->build(0);

    cudaDeviceSynchronize();

    dpds1.local()->coosvels.downloadFromDevice(0);
    dpds2.local()->coosvels.downloadFromDevice(0);


    // Set non-zero velocities
    for (int i=0; i<dpds1.local()->size(); i++)
        dpds1.local()->coosvels[i].u = length*make_float3(drand48() - 0.5, drand48() - 0.5, drand48() - 0.5);

    for (int i=0; i<dpds2.local()->size(); i++)
        dpds2.local()->coosvels[i].u = length*make_float3(drand48() - 0.5, drand48() - 0.5, drand48() - 0.5);


    dpds1.local()->coosvels.uploadToDevice(0);
    dpds2.local()->coosvels.uploadToDevice(0);


    std::vector<Particle> initial(np), rearranged(np);
    for (int i=0; i<np; i++)
    {
        initial[i] = ( i < dpds1.local()->size() ) ?
            dpds1.local()->coosvels[i] :
            dpds2.local()->coosvels[i - dpds1.local()->size()];
    }

    const float k = 1;
    const float dt = 0.002;
    const float kbT = 1.0f;
    const float gammadpd = 20;
    const float sigmadpd = sqrt(2 * gammadpd * kbT);
    const float sigma_dt = sigmadpd / sqrt(dt);
    const float adpd = 50;

    Pairwise_Norandom_DPD dpdInt(rc, adpd, gammadpd, kbT, dt, k);
    Interaction *inter = new InteractionPair<Pairwise_Norandom_DPD>("dpd", rc, dpdInt);

    PinnedBuffer<int> counter(1);

    for (int i=0; i<1; i++)
    {
        dpds1.local()->forces.clear(0);
        dpds2.local()->forces.clear(0);

        inter->regular(&dpds1, &dpds1, cells1, cells1, 0, 0);
        inter->regular(&dpds2, &dpds2, cells2, cells2, 0, 0);
        inter->regular(&dpds2, &dpds1, cells2, cells1, 0, 0);

        cudaDeviceSynchronize();
    }

    HostBuffer<Force> frcs1, frcs2;
    frcs1.copy(dpds1.local()->forces, 0);
    frcs2.copy(dpds2.local()->forces, 0);

    cudaDeviceSynchronize();

    std::vector<Force> hacc(np);
    for (int i=0; i<np; i++)
        hacc[i] = ( i < dpds1.local()->size() ) ?
            frcs1[i] :
            frcs2[i - dpds1.local()->size()];

    std::vector<int> hcellsstart(cells1->totcells+1);
    std::vector<int> hcellssize(cells1->totcells);
    std::vector<int> order(np);

    cudaDeviceSynchronize();

    fprintf(stderr, "finished, reducing acc\n");
    double3 a = {};
    for (int i=0; i<np; i++)
    {
        a.x += hacc[i].f.x;
        a.y += hacc[i].f.y;
        a.z += hacc[i].f.z;
    }
    fprintf(stderr, "Reduced acc: %e %e %e\n\n", a.x, a.y, a.z);

    fprintf(stderr, "Checking (this is not necessarily a cubic domain)......\n");

    makeCells(initial.data(), rearranged.data(), hcellsstart.data(), hcellssize.data(), order.data(), np, cells1->cellInfo());

    std::vector<Force> refAcc(hacc.size());

    auto addForce = [&](int dstId, int srcId, Force& a)
    {
        const float _xr = rearranged[dstId].r.x - rearranged[srcId].r.x;
        const float _yr = rearranged[dstId].r.y - rearranged[srcId].r.y;
        const float _zr = rearranged[dstId].r.z - rearranged[srcId].r.z;

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
        xr * (rearranged[dstId].u.x - rearranged[srcId].u.x) +
        yr * (rearranged[dstId].u.y - rearranged[srcId].u.y) +
        zr * (rearranged[dstId].u.z - rearranged[srcId].u.z);

        int sid = rearranged[srcId].i1;
        int did = rearranged[dstId].i1;
        const float myrandnr = ((min(sid, did) ^ max(sid, did)) % 13) - 6;

        const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigma_dt * myrandnr) * wr;

        a.f.x += strength * xr;
        a.f.y += strength * yr;
        a.f.z += strength * zr;
    };

#pragma omp parallel for collapse(3)
    for (int cx = 0; cx < cells1->ncells.x; cx++)
        for (int cy = 0; cy < cells1->ncells.y; cy++)
            for (int cz = 0; cz < cells1->ncells.z; cz++)
            {
                const int cid = cells1->encode(cx, cy, cz);

                const int2 start_size = make_int2(hcellsstart[cid], hcellssize[cid]);

                for (int dstId = start_size.x; dstId < start_size.x + start_size.y; dstId++)
                {
                    Force a {{0,0,0},0};

                    for (int dx = -1; dx <= 1; dx++)
                        for (int dy = -1; dy <= 1; dy++)
                            for (int dz = -1; dz <= 1; dz++)
                            {
                                const int srcCid = cells1->encode(cx+dx, cy+dy, cz+dz);
                                if (srcCid >= cells1->totcells || srcCid < 0) continue;

                                const int2 srcStart_size = make_int2(hcellsstart[srcCid], hcellssize[srcCid]);

                                for (int srcId = srcStart_size.x; srcId < srcStart_size.x + srcStart_size.y; srcId++)
                                {
                                    if (dstId != srcId)
                                        addForce(dstId, srcId, a);
                                }
                            }

                    refAcc[dstId].f.x = a.f.x;
                    refAcc[dstId].f.y = a.f.y;
                    refAcc[dstId].f.z = a.f.z;
                }
            }

    double l2 = 0, linf = -1;

    std::vector<Force> finalFrcs(np);
    for (int i=0; i<np; i++)
    {
        finalFrcs[order[i]] = refAcc[i];
    }

    for (int i=0; i<np; i++)
    {
        double perr = -1;

        double toterr = 0;
        for (int c=0; c<3; c++)
        {
            double err;
            if (c==0) err = fabs(finalFrcs[i].f.x - hacc[i].f.x);
            if (c==1) err = fabs(finalFrcs[i].f.y - hacc[i].f.y);
            if (c==2) err = fabs(finalFrcs[i].f.z - hacc[i].f.z);

            toterr += err;
            linf = max(linf, err);
            perr = max(perr, err);
            l2 += err * err;
        }

        if (argc > 1 && (perr > 0.1 || std::isnan(toterr)))
        {
            fprintf(stderr, "id %d (%d),  %12f %12f %12f     ref %12f %12f %12f    diff   %12f %12f %12f\n", i, (initial[i].i1),
                    hacc[i].f.x, hacc[i].f.y, hacc[i].f.z,
                    finalFrcs[i].f.x, finalFrcs[i].f.y, finalFrcs[i].f.z,
                    hacc[i].f.x-finalFrcs[i].f.x, hacc[i].f.y-finalFrcs[i].f.y, hacc[i].f.z-finalFrcs[i].f.z);
        }
    }


    l2 = sqrt(l2 / np);
    fprintf(stderr, "L2   norm: %f\n", l2);
    fprintf(stderr, "Linf norm: %f\n", linf);

    if (linf < 0.02)
        printf("Success!\n");
    else
        printf("Failed\n");

    CUDA_Check( cudaPeekAtLastError() );
    return 0;
}
