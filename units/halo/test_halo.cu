// Yo ho ho ho
#define private   public
#define protected public

#include <core/pvs/particle_vector.h>
#include <core/celllist.h>
#include <core/mpi/api.h>
#include <core/logger.h>

#include <core/containers.h>

#include <core/initial_conditions/uniform_ic.h>

Logger logger;

Particle addShift(Particle p, float a, float b, float c)
{
    Particle res = p;
    res.r.x += a;
    res.r.y += b;
    res.r.z += c;

    return res;
}

int main(int argc, char ** argv)
{
    // Init
    bool verbose = argc > 1;

    int nranks, rank;
    int ranks[] = {1, 1, 1};
    int periods[] = {1, 1, 1};
    MPI_Comm cartComm;

    MPI_Init(&argc, &argv);
    logger.init(MPI_COMM_WORLD, "halo.log", 9);

    MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
    MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
    MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

    float3 length{80,70,55};
    float3 domainStart = -length / 2.0f;
    const float rc = 1.0f;
    ParticleVector dpds("dpd", 1.0f);
    PrimaryCellList cells(&dpds, rc, length);

    DomainInfo domain{length, domainStart, length};

    InitialConditions* ic = new UniformIC(8.0);
    ic->exec(MPI_COMM_WORLD, &dpds, domain,  0);

    cells.build(0);

    dpds.local()->coosvels.downloadFromDevice(0);

    ParticleHaloExchanger halo(cartComm);
    halo.attach(&dpds, &cells);

    cells.build(0);
    CUDA_Check( cudaStreamSynchronize(0) );

    for (int i=0; i<10; i++)
    {
        halo.init(0);
        halo.finalize(0);

        dpds.haloValid = false;
    }

    std::vector<Particle> bufs[27];
    dpds.local()->coosvels.downloadFromDevice(0);
    dpds.halo()->coosvels.downloadFromDevice(0);

    for (int i=0; i<dpds.local()->size(); i++)
    {
        Particle& p = dpds.local()->coosvels[i];

        int3 code = cells.getCellIdAlongAxes(p.r);
        int cx = code.x,  cy = code.y,  cz = code.z;
        auto ncells = cells.ncells;

        // 6
        if (cx == 0)          bufs[ (1*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0,         0));
        if (cx == ncells.x-1) bufs[ (1*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0,         0));
        if (cy == 0)          bufs[ (1*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y,         0));
        if (cy == ncells.y-1) bufs[ (1*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y,         0));
        if (cz == 0)          bufs[ (0*3 + 1)*3 + 1 ].push_back(addShift(p,         0,         0,  length.z));
        if (cz == ncells.z-1) bufs[ (2*3 + 1)*3 + 1 ].push_back(addShift(p,         0,         0, -length.z));

        // 12
        if (cx == 0          && cy == 0)          bufs[ (1*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y,         0));
        if (cx == ncells.x-1 && cy == 0)          bufs[ (1*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y,         0));
        if (cx == 0          && cy == ncells.y-1) bufs[ (1*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y,         0));
        if (cx == ncells.x-1 && cy == ncells.y-1) bufs[ (1*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y,         0));

        if (cy == 0          && cz == 0)          bufs[ (0*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y,  length.z));
        if (cy == ncells.y-1 && cz == 0)          bufs[ (0*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y,  length.z));
        if (cy == 0          && cz == ncells.z-1) bufs[ (2*3 + 0)*3 + 1 ].push_back(addShift(p,         0,  length.y, -length.z));
        if (cy == ncells.y-1 && cz == ncells.z-1) bufs[ (2*3 + 2)*3 + 1 ].push_back(addShift(p,         0, -length.y, -length.z));


        if (cz == 0          && cx == 0)          bufs[ (0*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0,  length.z));
        if (cz == ncells.z-1 && cx == 0)          bufs[ (2*3 + 1)*3 + 0 ].push_back(addShift(p,  length.x,         0, -length.z));
        if (cz == 0          && cx == ncells.x-1) bufs[ (0*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0,  length.z));
        if (cz == ncells.z-1 && cx == ncells.x-1) bufs[ (2*3 + 1)*3 + 2 ].push_back(addShift(p, -length.x,         0, -length.z));

        // 8
        if (cx == 0          && cy == 0          && cz == 0)          bufs[ (0*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y,  length.z));
        if (cx == 0          && cy == 0          && cz == ncells.z-1) bufs[ (2*3 + 0)*3 + 0 ].push_back(addShift(p,  length.x,  length.y, -length.z));
        if (cx == 0          && cy == ncells.y-1 && cz == 0)          bufs[ (0*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y,  length.z));
        if (cx == 0          && cy == ncells.y-1 && cz == ncells.z-1) bufs[ (2*3 + 2)*3 + 0 ].push_back(addShift(p,  length.x, -length.y, -length.z));
        if (cx == ncells.x-1 && cy == 0          && cz == 0)          bufs[ (0*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y,  length.z));
        if (cx == ncells.x-1 && cy == 0          && cz == ncells.z-1) bufs[ (2*3 + 0)*3 + 2 ].push_back(addShift(p, -length.x,  length.y, -length.z));
        if (cx == ncells.x-1 && cy == ncells.y-1 && cz == 0)          bufs[ (0*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y,  length.z));
        if (cx == ncells.x-1 && cy == ncells.y-1 && cz == ncells.z-1) bufs[ (2*3 + 2)*3 + 2 ].push_back(addShift(p, -length.x, -length.y, -length.z));
    }

    auto& helper = halo.helpers[0];

    bool success = true;
    for (int i = 0; i<27; i++)
    {
        std::sort(bufs[i].begin(), bufs[i].end(), [] (Particle& a, Particle& b) { return a.i1 < b.i1; });

        std::sort((Particle*)helper->sendBuf.hostPtr() + helper->sendOffsets[i],
                (Particle*)helper->sendBuf.hostPtr() + helper->sendOffsets[i+1], [] (Particle& a, Particle& b) { return a.i1 < b.i1; });


        if (bufs[i].size() != helper->sendSizes[i])
        {
            if (verbose)
                printf("%2d-th halo differs in size: %5d, expected %5d\n", i, helper->sendSizes[i], (int)bufs[i].size());
            
            success = false;
        }
        else
        {
            auto ptr = (Particle*)helper->sendBuf.hostPtr() + helper->sendOffsets[i];
            for (int pid = 0; pid < helper->sendSizes[i]; pid++)
            {
                const float diff = std::max({
                    fabs(ptr[pid].r.x - bufs[i][pid].r.x),
                    fabs(ptr[pid].r.y - bufs[i][pid].r.y),
                    fabs(ptr[pid].r.z - bufs[i][pid].r.z) });

                if (bufs[i][pid].i1 != ptr[pid].i1 || diff > 1e-5)
                {
                    success = false;
                    
                    if (verbose)
                        printf("Halo %2d:  %5d [%10.3e %10.3e %10.3e], expected %5d [%10.3e %10.3e %10.3e]\n",
                                i, ptr[pid].i1, ptr[pid].r.x, ptr[pid].r.y, ptr[pid].r.z,
                                bufs[i][pid].i1, bufs[i][pid].r.x, bufs[i][pid].r.y, bufs[i][pid].r.z);
                }
            }
        }
    }
    
    if (success)
        printf("Success!\n");
    else
    {
        printf("Failed!\n");
        exit(1);
    }
        
    return 0;
}
