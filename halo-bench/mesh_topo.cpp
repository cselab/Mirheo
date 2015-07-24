
/**
 * Copyright (c) 2012-2013 Los Alamos National Security, LLC.
 *                         All rights reserved.
 *
 * This program was prepared by Los Alamos National Security, LLC at Los Alamos
 * National Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S.
 * Department of Energy (DOE). All rights in the program are reserved by the DOE
 * and Los Alamos National Security, LLC. Permission is granted to the public to
 * copy and use this software without charge, provided that this Notice and any
 * statement of authorship are reproduced on all copies. Neither the U.S.
 * Government nor LANS makes any warranty, express or implied, or assumes any
 * liability or responsibility for the use of this software.
 */

/* simple example application that uses rca to get mesh coordinates */

/* @author samuel k. gutierrez - samuel@lanl.gov */

/* to run:
 * aprun -n <X> rca-mesh-coords
 * aprun -n <X> -N <Y> rca-mesh-coords
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
#include <pmi.h>

enum {
    SUCCESS = 0,
    FAILURE
};

static int print_all_mesh_coords(int size)
{
    int i, nid;
    pmi_mesh_coord_t xyz;

    pmi_mesh_coord_t maxdim;
    PMI_Get_max_dimension(&maxdim);
    FILE *fp = fopen("topo.txt", "w");

    printf("max coords - (%d, %d, %d)\n", maxdim.mesh_x+1, maxdim.mesh_y+1, maxdim.mesh_z+1);
    fprintf(fp, "    %3d %3d %3d\n", maxdim.mesh_x+1, maxdim.mesh_y+1, maxdim.mesh_z+1);

    for (i = 0; i < size; ++i) {
        if (PMI_SUCCESS != PMI_Get_nid(i, &nid)) {
            fprintf(stderr, "PMI_Get_nid failure\n");
            return FAILURE;
        }
        PMI_Get_meshcoord(nid, &xyz);
        printf("%d nid %d coords - (%d, %d, %d)\n", i, nid, xyz.mesh_x, xyz.mesh_y, xyz.mesh_z);

        fprintf(fp, "%3d %3d %3d %3d\n", i, xyz.mesh_x, xyz.mesh_y, xyz.mesh_z);

    }
    fclose(fp);

    return SUCCESS;
}

int main(int argc, char **argv)
{
	int rc;
	int rank, size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		if (SUCCESS != (rc = print_all_mesh_coords(size))) {
			fprintf(stderr, "print_all_mesh_coords failure -- "
				"cannot continue...\n");
			goto out;
		}
	}

out:
	MPI_Finalize();
	return (rc == SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE;
}
