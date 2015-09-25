/*
 *  dumper.cu
 *  ctc daint
 *
 *  Created by Dmitry Alexeev on Sep 24, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <vector>
#include <sys/stat.h>

#include "dumper.h"
#include "containers.h"
#include "ctc.h"
#include "io.h"

using namespace std;

Dumper::Dumper(MPI_Comm iocomm, MPI_Comm iocartcomm, MPI_Comm intercomm) : iocomm(iocomm), iocartcomm(iocartcomm), intercomm(intercomm)
{
    CollectionRBC *rdummy = new CollectionRBC(iocartcomm);
    CollectionCTC *cdummy = new CollectionCTC(iocartcomm);
    nrbcverts = rdummy->get_nvertices();
    nctcverts = cdummy->get_nvertices();

    MPI_CHECK(MPI_Comm_rank(iocomm, &rank));
}

void Dumper::qoi(Particle* rbcs, Particle * ctcs, int nrbcparts, int nctcparts, const float tm)
{
    int dims[3], periods[3], coords[3];
    MPI_CHECK( MPI_Cart_get(iocartcomm, 3, dims, periods, coords) );

    const int subdomain[3] = {XSIZE_SUBDOMAIN, YSIZE_SUBDOMAIN, ZSIZE_SUBDOMAIN};
    const int nbins = 15; // number of rows

    vector<int> locRBChisto(nbins, 0);
    vector<int> locCTChisto(nbins, 0);
    float totcom[3] = {0, 0, 0};

    const float rwidth = 56;
    const float offset = -40;
    const float tan_a  = tan(1.7 / 180.0 * M_PI);

    const int nrbcs = nrbcparts / nrbcverts;
    const int nctcs = nctcparts / nctcverts;

    if (nrbcs)
    {
        for (int p = 0; p < nrbcs; p++)
        {
            float com[3] = {0, 0, 0};
            Particle * cur = rbcs + p * nrbcverts;

            for (int i=0; i < nrbcverts; i++)
                for (int d = 0; d<3; d++)
                {
                    totcom[d] += cur[i].x[d] + (coords[d] + 0.5) * subdomain[d];
                    com[d]    += cur[i].x[d];
                }

            for (int d = 0; d<3; d++)
                com[d] = com[d] / nrbcverts + (coords[d] + 0.5) * subdomain[d];

            int irow = floor( (com[1] - offset - com[0] * tan_a) / rwidth );
            if (irow >= nbins) irow = nbins - 1;
            if (irow < 0) irow = 0;

            locRBChisto[irow]++;
        }
    }

    if (nctcs)
        for (int p = 0; p < nctcs; p++)
        {
            float com[3] = {0, 0, 0};
            Particle * cur = ctcs + p * nctcverts;

            for (int i=0; i < nctcverts; i++)
                for (int d = 0; d<3; d++)
                    com[d] += cur[i].x[d];

            for (int d = 0; d<3; d++)
                com[d] = com[d] / nctcverts + (coords[d] + 0.5) * subdomain[d];

            int irow = floor( (com[1] - offset - com[0] * tan_a) / rwidth );
            if (irow >= nbins) irow = nbins - 1;
            if (irow < 0) irow = 0;

            locCTChisto[irow]++;
        }

    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &locRBChisto[0], &locRBChisto[0], nbins, MPI_INT, MPI_SUM, 0, iocomm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &locCTChisto[0], &locCTChisto[0], nbins, MPI_INT, MPI_SUM, 0, iocomm) );
    MPI_CHECK( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : totcom, totcom, 3, MPI_FLOAT, MPI_SUM, 0, iocomm) );


    if (nctcs)
    {
        float com[3] = {0, 0, 0};

        Particle * cur = ctcs;
        for (int i=0; i < nctcverts; i++)
            for (int d = 0; d<3; d++)
                com[d] += cur[i].x[d];

        for (int d = 0; d<3; d++)
            com[d] = com[d] / nctcverts + (coords[d] + 0.5) * subdomain[d];

        FILE* f = fopen("ctccom.txt", qoiid == 0 ? "w" : "a");
        fprintf(f, "%f   %e %e %e\n", tm, com[0], com[1], com[2]);
        fclose(f);
    }

    if (rank == 0)
    {
        if (nrbcs)
        {
            FILE* fout = fopen("rbchisto.dat", qoiid == 0 ? "w" : "a");
            fprintf(fout, "\n %f\n", tm);
            for (int i=0; i<nbins; i++)
                fprintf(fout, "%d   %d\n", i, locRBChisto[i]);
            fclose(fout);
        }

        if (nctcs)
        {
            FILE* fout = fopen("ctchisto.dat", qoiid == 0 ? "w" : "a");
            fprintf(fout, "\n %f\n", tm);
            for (int i=0; i<nbins; i++)
                fprintf(fout, "%d   %d\n", i, locCTChisto[i]);
            fclose(fout);
        }

        if (nrbcs)
        {
            float totrbcs = 0;
            for (int i=0; i<nbins; i++)
                totrbcs += locRBChisto[i];
            totrbcs *= nrbcverts;

            FILE* fout = fopen("rbccom.txt", qoiid == 0 ? "w" : "a");
            fprintf(fout, "%f   %e %e %e\n", tm, totcom[0] / totrbcs, totcom[1] / totrbcs, totcom[2] / totrbcs);
            fclose(fout);
        }
    }
    qoiid++;
}


void Dumper::do_dump()
{
    int nparticles, nrbcparts, nctcparts;

    int iddatadump = 0;
    bool wallcreated = false;

    MPI_Status status;

    while (1)
    {
        //printf("Waiting for data...\n");
        MPI_CHECK( MPI_Recv(&nparticles, 1, MPI_INT, rank, 0, intercomm, &status) );
        MPI_CHECK( MPI_Recv(&nrbcparts,  1, MPI_INT, rank, 0, intercomm, &status) );
        MPI_CHECK( MPI_Recv(&nctcparts,  1, MPI_INT, rank, 0, intercomm, &status) );

        int n = nparticles + nrbcparts + nctcparts;
        //printf("Received %d (%d + %d + %d) particles\n", n, nparticles, nrbcparts, nctcparts);

        if (n < 0) return;
        if (n > particles.size()) particles.resize(1.1*n);
        if (n > accelerations.size()) accelerations.resize(1.1*n);

        Particle* p    = &particles[0];
        Acceleration* a = &accelerations[0];
        MPI_CHECK( MPI_Recv(p, n, Particle::datatype(),     rank, 0, intercomm, &status) );
        MPI_CHECK( MPI_Recv(a, n, Acceleration::datatype(), rank, 0, intercomm, &status) );

        H5PartDump dump_part("allparticles->h5part", iocomm, iocartcomm), *dump_part_solvent = NULL;
        H5FieldDump dump_field(iocartcomm);


        if (rank == 0)
            mkdir("xyz", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        MPI_CHECK(MPI_Barrier(iocomm));

        {
            NVTX_RANGE("diagnostics", NVTX_C1);
            diagnostics(iocomm, iocartcomm, p, n, dt, iddatadump, a);
        }

        if (xyz_dumps)
        {
            NVTX_RANGE("xyz dump", NVTX_C2);

            if (walls && iddatadump >= wall_creation_stepid && !wallcreated)
            {
                if (rank == 0)
                {
                    if( access("xyz/particles-equilibration.xyz", F_OK ) == -1 )
                        rename ("xyz/particles.xyz", "xyz/particles-equilibration.xyz");

                    if( access( "xyz/rbcs-equilibration.xyz", F_OK ) == -1 )
                        rename ("xyz/rbcs.xyz", "xyz/rbcs-equilibration.xyz");

                    if( access( "xyz/ctcs-equilibration.xyz", F_OK ) == -1 )
                        rename ("xyz/ctcs.xyz", "xyz/ctcs-equilibration.xyz");
                }

                MPI_CHECK(MPI_Barrier(iocomm));

                wallcreated = true;
            }

            xyz_dump(iocomm, iocartcomm, "xyz/particles->xyz", "all-particles", p, n, iddatadump > 0);
        }

        if (hdf5part_dumps)
        {
            if (!dump_part_solvent && walls && iddatadump >= wall_creation_stepid)
            {
                dump_part.close();

                dump_part_solvent = new H5PartDump("solvent-particles->h5part", iocomm, iocartcomm);
            }

            if (dump_part_solvent)
                dump_part_solvent->dump(p, n);
            else
                dump_part.dump(p, n);
        }

        if (hdf5field_dumps)
        {
            dump_field.dump(iocomm, p, nparticles, iddatadump);
        }

        {
            if (rbcs)
                CollectionRBC::dump(iocomm, iocartcomm, p + nparticles, a + nparticles, nrbcparts, iddatadump);

            if (ctcs)
                CollectionCTC::dump(iocomm, iocartcomm, p + nparticles + nrbcparts, a + nparticles + nrbcparts, nctcparts, iddatadump);
        }

        qoi(p + nparticles, p + nparticles + nrbcparts, nrbcparts, nctcparts, iddatadump * dt);

        ++iddatadump;
    }
}

