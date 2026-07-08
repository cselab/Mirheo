#define private   public
#define protected public

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cassert>
#include <algorithm>

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/initial_conditions/uniform.h>

#include <gtest/gtest.h>

using namespace mirheo;

bool verbose = false;

void test_domain(real3 length, real rc, real density, int nbuilds)
{
    bool success = true;
    DomainInfo domain{length, {0,0,0}, length};
    real dt = 0; // dummy dt
    MirState state(domain, dt);

    ParticleVector dpds(&state, "dpd", 1.0f);
    std::unique_ptr<CellList> cells = std::make_unique<PrimaryCellList>(&dpds, rc, length);

    UniformIC ic(density);
    ic.exec(MPI_COMM_WORLD, &dpds, 0);

    const int np = dpds.local()->size();
    HostBuffer<real4> initialPos(np), initialVel(np);

    std::copy(dpds.local()->positions ().begin(), dpds.local()->positions ().end(), initialPos.begin());
    std::copy(dpds.local()->velocities().begin(), dpds.local()->velocities().end(), initialVel.begin());

    for (int i = 0; i < nbuilds; i++)
    {
        cells->build(defaultStream);
        dpds.cellListStamp++;
    }

    dpds.local()->positions ().downloadFromDevice(defaultStream, ContainersSynch::Asynch);
    dpds.local()->velocities().downloadFromDevice(defaultStream, ContainersSynch::Synch);

    HostBuffer<int> hcellsStart(cells->totcells+1);
    HostBuffer<int> hcellsSize (cells->totcells+1);

    hcellsStart.copy(cells->cellStarts, defaultStream);
    hcellsSize. copy(cells->cellSizes,  defaultStream);

    HostBuffer<int> cellscount(cells->totcells+1);
    for (int i = 0; i < cells->totcells+1; ++i)
        cellscount[i] = 0;

    int total = 0;
    for (size_t pid = 0; pid < initialPos.size(); ++pid)
    {
        auto coo = make_real3(initialPos[pid]);

        int actCid = cells->getCellId(coo);
        if (actCid >= 0)
        {
            cellscount[actCid]++;
            total++;
        }
    }

    if (verbose)
        printf("np = %d, vs reference  %d\n", dpds.local()->size(), total);

    for (int cid = 0; cid < cells->totcells+1; cid++)
        if ( (hcellsSize[cid]) != cellscount[cid] )
        {
            success = false;

            if (verbose)
                printf("cid %d:  %d (correct %d),  %d\n",
                        cid, hcellsSize[cid], cellscount[cid], hcellsStart[cid]);
        }

    auto& positions  = dpds.local()->positions();
    auto& velocities = dpds.local()->velocities();

    for (int cid = 0; cid < cells->totcells; cid++)
    {
        const int start = hcellsStart[cid];
        const int size = hcellsSize[cid];
        for (int pid = start; pid < start + size; pid++)
        {
            auto pDev = Particle(positions[pid], velocities[pid]);
            auto cooDev = pDev.r;
            auto velDev = pDev.u;
            const auto origId = pDev.getId();

            auto p = Particle(initialPos[origId], initialVel[origId]);
            auto coo = p.r;
            auto vel = p.u;

            const real diff = std::max({
                fabs(coo.x - cooDev.x), fabs(coo.y - cooDev.y), fabs(coo.z - cooDev.z),
                fabs(vel.x - velDev.x), fabs(vel.y - velDev.y), fabs(vel.z - velDev.z) });

            int actCid = cells->getCellId<CellListsProjection::NoClamp>(cooDev);

            if (cid != actCid || diff > 1e-5)
            {
                success = false;

                if (verbose)
                    printf("cid  %d,  correct cid  %d  for pid %d:  [%e %e %e  %ld]  correct: [%e %e %e  %ld]\n",
                            cid, actCid, pid, cooDev.x, cooDev.y, cooDev.z, origId,
                           coo.x, coo.y, coo.z, p.getId());
            }
        }
    }

    ASSERT_TRUE(success);
}


TEST (CELLLISTS, DomainVaries)
{
    real rc = 1.0, density = 7.5;
    int ncalls = 1;

    test_domain(make_real3(64, 64, 64), rc, density, ncalls);
    test_domain(make_real3(64, 32, 16), rc, density, ncalls);
}

TEST (CELLLISTS, rcVaries)
{
    real3 domain = make_real3(32, 32, 32);
    real density = 7.5;
    int ncalls = 1;

    test_domain(domain, 0.5, density, ncalls);
    test_domain(domain, 1.2, density, ncalls);
}

TEST (CELLLISTS, DensityVaries)
{
    real3 domain = make_real3(32, 32, 32);
    real rc = 1.0;
    int ncalls = 1;

    test_domain(domain, rc, 2.0, ncalls);
    test_domain(domain, rc, 8.0, ncalls);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "cells.log", 9);

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
