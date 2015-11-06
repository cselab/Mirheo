/*
 *  osu_latency_rdp.cpp
 *  Part of uDeviceX/halo-bench/
 *
 *  Created and authored by phadjido 2015-03-12 on 18:59:02.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#define BENCHMARK "OSU MPI%s Latency Test"
/*
 * Copyright (C) 2002-2014 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */

#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "hpm.cpp" 

#define MESSAGE_ALIGNMENT 64
#define MAX_ALIGNMENT 65536
//#define MAX_MSG_SIZE (1<<22)
#define MAX_MSG_SIZE (1<<20)
#define MYBUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

#define LOOP_LARGE  100
#define SKIP_LARGE  10
#define LARGE_MESSAGE_SIZE  8192

int skip = 1000;
int loop = 100000;



int root_id = 0;

#if 1
enum {
    XSIZE_SUBDOMAIN = 48,
    YSIZE_SUBDOMAIN = 48,
    ZSIZE_SUBDOMAIN = 48,

    XMARGIN_WALL = 6,
    YMARGIN_WALL = 6,
    ZMARGIN_WALL = 6,
};

const int numberdensity = 4*6;
float safety_factor = 1.2;

int send_counts[26];
int Ncnt = 1;

void init_counts(int myid)
{
	int i;
	for(i = 1; i < 27; ++i)
	{
		const int d[3] = { (i + 1) % 3 - 1, (i / 3 + 1) % 3 - 1, (i / 9 + 1) % 3 - 1 };

		const int nhalodir[3] =  {
			d[0] != 0 ? 1 : XSIZE_SUBDOMAIN,
			d[1] != 0 ? 1 : YSIZE_SUBDOMAIN,
			d[2] != 0 ? 1 : ZSIZE_SUBDOMAIN
		};

		const int nhalocells = nhalodir[0] * nhalodir[1] * nhalodir[2];
		const int estimate = numberdensity * safety_factor * nhalocells;
		send_counts[i-1] = estimate - 4;
	}

	int max_cnt = -1;
	for (i = 0; i < 26; i++) {
		if (send_counts[i] > max_cnt) max_cnt = send_counts[i];
	}
	Ncnt = max_cnt;

	if (myid == 0)
	{
		printf("send_counts: ");
		for (i = 0; i < 26; i++) printf("%d ", send_counts[i]);
		printf("\n");
		fflush(0);
	}
}



#endif

int main (int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    float **s_buf, **r_buf;
    double t_start = 0.0, t_end = 0.0;

    if (argc == 2) {
	root_id = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if(myid == root_id) {
         fprintf(stderr, "Running test with root id %d and %d proceses\n", root_id, numprocs);
	fflush(0);
    }

    init_counts(myid);

    s_buf = (float **)malloc(numprocs*sizeof(float *));
    r_buf = (float **)malloc(numprocs*sizeof(float *));
    if (myid == root_id) { 
	for (i = 1; i < numprocs; i++) {
		s_buf[i] = (float *)calloc(1, send_counts[(i-1)%26]*sizeof(float));
		r_buf[i] = (float *)calloc(1, send_counts[(i-1)%26]*sizeof(float));
	}
    }
    else {
	s_buf[0] = (float *)calloc(1, send_counts[(myid-1)%26]*sizeof(float));
	r_buf[0] = (float *)calloc(1, send_counts[(myid-1)%26]*sizeof(float));
    }


	loop = LOOP_LARGE;
	skip = SKIP_LARGE;

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Request req[2*numprocs];
	MPI_Status statuses[2*numprocs];

	HPM hpm;

	if(myid == root_id) {
		for(i = 0; i < loop + skip; i++) {
			if(i == skip) t_start = MPI_Wtime();
			int rq, r, n;
			if (i >= skip) hpm.HPM_Start("loop");

			rq = 0;
			for (n = 0; n < numprocs; n++) {
				if (n == root_id) continue;
				MPI_Irecv(r_buf[n], send_counts[(n-1)%26], MPI_FLOAT, n, 1, MPI_COMM_WORLD, &req[rq]);
				rq++;
			}

			for (n = 0; n < numprocs; n++) {
				if (n == root_id) continue;
				MPI_Isend(s_buf[n], send_counts[(n-1)%26], MPI_FLOAT, n, 1, MPI_COMM_WORLD, &req[rq]);
				rq++;
			}
			MPI_Waitall(rq, req, statuses);
			if (i >= skip) hpm.HPM_Stop("loop");

		}

		t_end = MPI_Wtime();
	}
	else if (myid != root_id) {
		for(i = 0; i < loop + skip; i++) {
			int r, rq;
		
			rq=0;
			MPI_Irecv(r_buf[0], send_counts[(myid-1)%26], MPI_FLOAT, root_id, 1, MPI_COMM_WORLD, &req[rq]);
			rq++;
#if 1
			MPI_Isend(s_buf[0], send_counts[(myid-1)%26], MPI_FLOAT, root_id, 1, MPI_COMM_WORLD, &req[rq]);
			rq++;
#else
			MPI_Send(s_buf[0], send_counts[(myid-1)%26], MPI_FLOAT, root_id, 1, MPI_COMM_WORLD);
#endif
			MPI_Waitall(rq, req, statuses);
		}
	}

	if(myid == root_id) {
		double latency = (t_end - t_start) * 1e6 / (1.0 * loop);
		fprintf(stdout, "latency = %f\n", latency);
		fflush(stdout);
		hpm.HPM_Stats();
	}

	MPI_Finalize();

	return 0;
}

