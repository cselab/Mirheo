// Yo ho ho ho
#define private   public
#define protected public

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cassert>
#include <algorithm>

#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/logger.h>
#include <core/initial_conditions/uniform.h>

#include <gtest/gtest.h>

Logger logger;
bool verbose = false;

void test_domain(float3 length, float rc, float density)
{
    bool success = true;
    float3 domainStart = -length / 2.0f;
    DomainInfo domain{length, {0,0,0}, length};
    float dt = 0; // dummy dt
    YmrState state(domain, dt);

    ParticleVector dpds(&state, "dpd", 1.0f);
    CellList *cells = new PrimaryCellList(&dpds, rc, length);

    UniformIC ic(density);
    ic.exec(MPI_COMM_WORLD, &dpds, 0);

    const int np = dpds.local()->size();
    HostBuffer<Particle> initial(np);
    auto initPtr = initial.hostPtr();
    for (int i=0; i<np; i++)
        initPtr[i] = dpds.local()->coosvels[i];

    for (int i=0; i<50; i++)
    {
        cells->build(0);
        dpds.cellListStamp++;
    }

    dpds.local()->coosvels.downloadFromDevice(0, ContainersSynch::Synch);

    HostBuffer<int> hcellsStart(cells->totcells+1);
    HostBuffer<int> hcellsSize(cells->totcells+1);

    hcellsStart.copy(cells->cellStarts, 0);
    hcellsSize. copy(cells->cellSizes, 0);

    HostBuffer<int> cellscount(cells->totcells+1);
    for (int i=0; i<cells->totcells+1; i++)
        cellscount[i] = 0;

    int total = 0;
    for (int pid=0; pid < initial.size(); pid++)
    {
        float3 coo{initial[pid].r.x, initial[pid].r.y, initial[pid].r.z};
        float3 vel{initial[pid].u.x, initial[pid].u.y, initial[pid].u.z};

        int actCid = cells->getCellId(coo);
        if (actCid >= 0)
        {
            cellscount[actCid]++;
            total++;
        }
    }

    if (verbose)
        printf("np = %d, vs reference  %d\n", dpds.local()->size(), total);
    for (int cid=0; cid < cells->totcells+1; cid++)
        if ( (hcellsSize[cid]) != cellscount[cid] )
        {
            success = false;
            
            if (verbose)
                printf("cid %d:  %d (correct %d),  %d\n",
                        cid, hcellsSize[cid], cellscount[cid], hcellsStart[cid]);
        }

    for (int cid=0; cid < cells->totcells; cid++)
    {
        const int start = hcellsStart[cid];
        const int size = hcellsSize[cid];
        for (int pid=start; pid < start + size; pid++)
        {
            const float3 cooDev{dpds.local()->coosvels[pid].r.x, dpds.local()->coosvels[pid].r.y, dpds.local()->coosvels[pid].r.z};
            const float3 velDev{dpds.local()->coosvels[pid].u.x, dpds.local()->coosvels[pid].u.y, dpds.local()->coosvels[pid].u.z};

            const int origId = dpds.local()->coosvels[pid].i1;

            float3 coo{initial[origId].r.x, initial[origId].r.y, initial[origId].r.z};
            float3 vel{initial[origId].u.x, initial[origId].u.y, initial[origId].u.z};

            const float diff = std::max({
                fabs(coo.x - cooDev.x), fabs(coo.y - cooDev.y), fabs(coo.z - cooDev.z),
                fabs(vel.x - velDev.x), fabs(vel.y - velDev.y), fabs(vel.z - velDev.z) });

            int actCid = cells->getCellId<CellListsProjection::NoClamp>(cooDev);

            if (cid != actCid || diff > 1e-5)
            {
                success = false;
                
                if (verbose)
                    printf("cid  %d,  correct cid  %d  for pid %d:  [%e %e %e  %d]  correct: [%e %e %e  %d]\n",
                            cid, actCid, pid, cooDev.x, cooDev.y, cooDev.z, dpds.local()->coosvels[pid].i1,
                            coo.x, coo.y, coo.z, initial[origId].i1);
            }
        }
    }
    
    ASSERT_TRUE(success);
}


TEST (CELLLISTS, DomainVaries)
{
    float rc = 1.0, density = 7.5;
    
    test_domain(make_float3(64, 64, 64), rc, density);
    test_domain(make_float3(64, 32, 16), rc, density);
}

TEST (CELLLISTS, rcVaries)
{
    float3 domain = make_float3(32, 32, 32);
    float density = 7.5;
    
    test_domain(domain, 0.5, density);
    test_domain(domain, 1.2, density);
}

TEST (CELLLISTS, DensityVaries)
{
    float3 domain = make_float3(32, 32, 32);
    float rc = 1.0;
    
    test_domain(domain, rc, 2.0);
    test_domain(domain, rc, 8.0);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    logger.init(MPI_COMM_WORLD, "cells.log", 9);
    
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
