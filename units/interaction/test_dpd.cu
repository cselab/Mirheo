#define protected public
#define private public

#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/containers.h>
#include <core/interactions/pairwise/impl.h>
#include <core/interactions/pairwise/kernels/norandom_dpd.h>
#include <core/initial_conditions/uniform.h>

#include <gtest/gtest.h>

#include <unistd.h>
#include <memory>

Logger logger;

bool verbose = false;

void makeCells(const float4 *inputPos, const float4 *inputVel,
               float4 *outputPos, float4 *outputVel,
               int *cellsStartSize, int *cellsSize,
               int *order, int np, CellListInfo cinfo)
{
    for (int i = 0; i < cinfo.totcells+1; i++)
        cellsSize[i] = 0;

    for (int i = 0; i < np; i++)
        cellsSize[cinfo.getCellId(make_float3(inputPos[i]))]++;

    cellsStartSize[0] = 0;
    for (int i = 1; i <= cinfo.totcells; i++)
        cellsStartSize[i] = cellsSize[i-1] + cellsStartSize[i-1];

    for (int i = 0; i < np; i++)
    {
        const int cid = cinfo.getCellId(make_float3(inputPos[i]));
        outputPos[cellsStartSize[cid]] = inputPos[i];
        outputVel[cellsStartSize[cid]] = inputVel[i];
        order[cellsStartSize[cid]] = i;

        cellsStartSize[cid]++;
    }

    for (int i = 0; i < cinfo.totcells; i++)
        cellsStartSize[i] -= cellsSize[i];
}

void execute(MPI_Comm comm, float3 length)
{
    DomainInfo domain{length, {0,0,0}, length};
    const float dt = 0.002;
    
    MirState state(domain, dt);

    const float rc = 1.0f;
    ParticleVector dpds1(&state, "dpd1", 1.0f);
    ParticleVector dpds2(&state, "dpd2", 1.0f);

    UniformIC ic1(4.5);
    UniformIC ic2(3.5);
    ic1.exec(comm, &dpds1, defaultStream);
    ic2.exec(comm, &dpds2, defaultStream);

    cudaDeviceSynchronize();

    const int np = dpds1.local()->size() + dpds2.local()->size();

    fprintf(stderr, " First part %d, second %d, total %d\n", dpds1.local()->size(), dpds2.local()->size(), np);

    std::unique_ptr<CellList> cells1 = std::make_unique<PrimaryCellList>(&dpds1, rc, length);
    std::unique_ptr<CellList> cells2 = std::make_unique<PrimaryCellList>(&dpds2, rc, length);
    cells1->build(defaultStream);
    cells2->build(defaultStream);

    cudaDeviceSynchronize();

    auto& pos1 = dpds1.local()->positions();
    auto& pos2 = dpds2.local()->positions();
    auto& vel1 = dpds1.local()->velocities();
    auto& vel2 = dpds2.local()->velocities();
    
    pos1.downloadFromDevice(defaultStream);
    pos2.downloadFromDevice(defaultStream);
    vel1.downloadFromDevice(defaultStream);
    vel2.downloadFromDevice(defaultStream);

    // Set non-zero velocities
    for (auto& v : vel1)
    {
        v.x = length.x * (drand48() - 0.5);
        v.y = length.y * (drand48() - 0.5);
        v.z = length.z * (drand48() - 0.5);
    }
    for (auto& v : vel2)
    {
        v.x = length.x * (drand48() - 0.5);
        v.y = length.y * (drand48() - 0.5);
        v.z = length.z * (drand48() - 0.5);
    }

    vel1.uploadToDevice(defaultStream);
    vel2.uploadToDevice(defaultStream);

    std::vector<float4> initialPos(np), initialVel(np), rearrangedPos(np), rearrangedVel(np);
    for (int i = 0; i < np; i++)
    {
        bool from1 = i < static_cast<int>(pos1.size());
        
        initialPos[i] = from1 ? pos1[i] : pos2[i-pos1.size()];
        initialVel[i] = from1 ? vel1[i] : vel2[i-vel1.size()];
    }

    const float k = 1;    
    const float kbT = 1.0f;
    const float gammadpd = 20;
    const float sigmadpd = sqrt(2 * gammadpd * kbT);
    const float sigma_dt = sigmadpd / sqrt(dt);
    const float adpd = 50;

    PairwiseNorandomDPD dpdInt(rc, adpd, gammadpd, kbT, dt, k);
    std::unique_ptr<Interaction> inter = std::make_unique<PairwiseInteractionImpl<PairwiseNorandomDPD>>(&state, "dpd", rc, dpdInt);

    PinnedBuffer<int> counter(1);

    for (int i = 0; i < 1; i++)
    {
        dpds1.local()->forces().clear(defaultStream);
        dpds2.local()->forces().clear(defaultStream);

        inter->local(&dpds1, &dpds1, cells1.get(), cells1.get(), defaultStream);
        inter->local(&dpds2, &dpds2, cells2.get(), cells2.get(), defaultStream);
        inter->local(&dpds2, &dpds1, cells2.get(), cells1.get(), defaultStream);

        cudaDeviceSynchronize();
    }

    HostBuffer<Force> frcs1, frcs2;
    frcs1.copy(dpds1.local()->forces(), defaultStream);
    frcs2.copy(dpds2.local()->forces(), defaultStream);

    cudaDeviceSynchronize();

    std::vector<Force> hacc(np);
    for (int i = 0; i < np; i++)
        hacc[i] = ( i < dpds1.local()->size() ) ?
            frcs1[i] :
            frcs2[i - dpds1.local()->size()];

    std::vector<int> hcellsstart(cells1->totcells+1);
    std::vector<int> hcellssize(cells1->totcells);
    std::vector<int> order(np);

    cudaDeviceSynchronize();

    fprintf(stderr, "finished, reducing acc\n");
    double3 a = {};
    for (int i = 0; i < np; i++)
    {
        a.x += hacc[i].f.x;
        a.y += hacc[i].f.y;
        a.z += hacc[i].f.z;
    }
    fprintf(stderr, "Reduced acc: %e %e %e\n\n", a.x, a.y, a.z);

    fprintf(stderr, "Checking (this is not necessarily a cubic domain)......\n");

    makeCells(initialPos.data(), initialVel.data(),
              rearrangedPos.data(), rearrangedVel.data(),
              hcellsstart.data(), hcellssize.data(),
              order.data(), np, cells1->cellInfo());

    std::vector<Force> refAcc(hacc.size());

    auto addForce = [&](int dstId, int srcId, Force& a)
    {
        Particle pdst(rearrangedPos[dstId], rearrangedVel[dstId]);
        Particle psrc(rearrangedPos[srcId], rearrangedVel[srcId]);
        const float _xr = pdst.r.x - psrc.r.x;
        const float _yr = pdst.r.y - psrc.r.y;
        const float _zr = pdst.r.z - psrc.r.z;

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
        xr * (pdst.u.x - psrc.u.x) +
        yr * (pdst.u.y - psrc.u.y) +
        zr * (pdst.u.z - psrc.u.z);

        int sid = psrc.i1;
        int did = pdst.i1;
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
    for (int i = 0; i < np; i++)
    {
        finalFrcs[order[i]] = refAcc[i];
    }

    for (int i = 0; i < np; i++)
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

        if (verbose && (perr > 0.1 || std::isnan(toterr)))
        {
            Particle pinitial(initialPos[i], initialVel[i]);
            fprintf(stderr, "id %d (%d),  %12f %12f %12f     ref %12f %12f %12f    diff   %12f %12f %12f\n", i, pinitial.i1,
                    hacc[i].f.x, hacc[i].f.y, hacc[i].f.z,
                    finalFrcs[i].f.x, finalFrcs[i].f.y, finalFrcs[i].f.z,
                    hacc[i].f.x-finalFrcs[i].f.x, hacc[i].f.y-finalFrcs[i].f.y, hacc[i].f.z-finalFrcs[i].f.z);
        }
    }


    l2 = sqrt(l2 / np);
    fprintf(stderr, "L2   norm: %f\n", l2);
    fprintf(stderr, "Linf norm: %f\n", linf);

    CUDA_Check( cudaPeekAtLastError() );

    ASSERT_LE(linf, 0.002);
    ASSERT_LE(l2,   0.002);
}

TEST(Interactions, smallDomain)
{
    float3 length{3, 4, 5};
    execute(MPI_COMM_WORLD, length);
}

TEST(Interactions, largerDomain)
{
    float3 length{80, 64, 29};
    execute(MPI_COMM_WORLD, length);
}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    logger.init(MPI_COMM_WORLD, "dpd.log", 9);

    testing::InitGoogleTest(&argc, argv);
    auto ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return ret;
}
