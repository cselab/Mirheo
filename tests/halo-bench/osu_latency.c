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

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#endif

#ifdef _ENABLE_CUDA_
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

#define MESSAGE_ALIGNMENT 64
#define MAX_ALIGNMENT 65536
//#define MAX_MSG_SIZE (1<<22)
#define MAX_MSG_SIZE (1<<20)
#define MYBUFSIZE (MAX_MSG_SIZE + MAX_ALIGNMENT)

#define LOOP_LARGE  100
#define SKIP_LARGE  10
#define LARGE_MESSAGE_SIZE  8192

#ifdef _ENABLE_OPENACC_
#   define OPENACC_ENABLED 1
#else
#   define OPENACC_ENABLED 0
#endif

#ifdef _ENABLE_CUDA_
#   define CUDA_ENABLED 1
#else
#   define CUDA_ENABLED 0
#endif

char s_buf_original[MYBUFSIZE];
char r_buf_original[MYBUFSIZE];

int skip = 1000;
int loop = 100000;

#ifdef _ENABLE_CUDA_
CUcontext cuContext;
#endif

enum po_ret_type {
    po_cuda_not_avail,
    po_openacc_not_avail,
    po_bad_usage,
    po_help_message,
    po_okay,
};

enum accel_type {
    none,
    cuda,
    openacc
};

struct options_s {
    char src;
    char dst;
    enum accel_type accel;
} options;

void usage (void);
int process_options (int argc, char *argv[]);
int allocate_memory (char **sbuf, char **rbuf, int rank);
void print_header (int rank);
void touch_data (void *sbuf, void *rbuf, int rank, size_t size);
void free_memory (void *sbuf, void *rbuf, int rank);
int init_accel (void);
int cleanup_accel (void);

int root_id = 0;

int main (int argc, char *argv[])
{
    int myid, numprocs, i;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0;
    int po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
           fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    if (argc == 2) {
	root_id = atoi(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (root_id == myid) {
        switch (po_ret) {
            case po_cuda_not_avail:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case po_openacc_not_avail:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case po_bad_usage:
            case po_help_message:
                usage();
                break;
        }
    }

    switch (po_ret) {
        case po_cuda_not_avail:
        case po_openacc_not_avail:
        case po_bad_usage:
            MPI_Finalize();
            exit(EXIT_FAILURE);
        case po_help_message:
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(numprocs != 2) {
        if(myid == root_id) {
            fprintf(stderr, "This test requires exactly two processes\n");
            fprintf(stderr, "Running test with root id %d and %d proceses\n", root_id, numprocs);
        }

        //MPI_Finalize();
        //exit(EXIT_FAILURE);
    }

    if (allocate_memory(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    print_header(myid);

    /* Latency test */
    for(size = 0; size <= MAX_MSG_SIZE; size = (size ? size * 2 : 1)) {
        touch_data(s_buf, r_buf, myid, size);

        if(size > LARGE_MESSAGE_SIZE) {
            loop = LOOP_LARGE;
            skip = SKIP_LARGE;
        }

        MPI_Barrier(MPI_COMM_WORLD);

	#define NNB 1

	MPI_Request req[2*NNB*numprocs];
	MPI_Status statuses[2*NNB*numprocs];

        if(myid == root_id) {
            for(i = 0; i < loop + skip; i++) {
                if(i == skip) t_start = MPI_Wtime();
		int rq;
		int r;
		int n;
#if 1
		rq = 0;
		for (n = 0; n < numprocs; n++) {
		if (n == root_id) continue;
		for (r = 0; r < NNB; r++) {
	                MPI_Irecv(r_buf, size, MPI_CHAR, n, 1, MPI_COMM_WORLD, &req[rq]);
			rq++;
		}
		}

		for (n = 0; n < numprocs; n++) {
		if (n == root_id) continue;
		for (r = NNB; r < 2*NNB; r++) {
#if 1
                	MPI_Isend(s_buf, size, MPI_CHAR, n, 1, MPI_COMM_WORLD, &req[rq]);
			rq++;
#else
                	MPI_Send(s_buf, size, MPI_CHAR, n, 1, MPI_COMM_WORLD);
#endif
		}
		}
//                MPI_Waitall(2*NNB, req, statuses);
//                MPI_Waitall(2*NNB*(numprocs-1), req, statuses);
                MPI_Waitall(rq, req, statuses);
#else
		for (r = 0; r < NNB; r++) {
                MPI_Send(s_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
                MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &reqstat);
		}
#endif
            }

            t_end = MPI_Wtime();
        }

        else if(myid != root_id) {
            for(i = 0; i < loop + skip; i++) {
		int r;
		int rq=0;
#if 1
		for (r = 0; r < NNB; r++) {
                 	MPI_Irecv(r_buf, size, MPI_CHAR, root_id, 1, MPI_COMM_WORLD, &req[rq]);
			rq++;
		}
		for (r = 0; r < NNB; r++) {
#if 1
                 	MPI_Isend(s_buf, size, MPI_CHAR, root_id, 1, MPI_COMM_WORLD, &req[rq]);
			rq++;
#else
                 	MPI_Send(s_buf, size, MPI_CHAR, root_id, 1, MPI_COMM_WORLD);
#endif
		}
//                MPI_Waitall(2*NNB, req, statuses);
                MPI_Waitall(rq, req, statuses);
#else
		for (r = 0; r < NNB; r++) {
	               MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &reqstat);
  	    	       MPI_Send(s_buf, size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
		}
#endif
            }
        }

        if(myid == root_id) {
//          double latency = (t_end - t_start) * 1e6 / (2.0 * loop);
            double latency = (t_end - t_start) * 1e6 / (1.0 * loop);

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);
    MPI_Finalize();

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

void
usage (void)
{
    if (CUDA_ENABLED || OPENACC_ENABLED) {
        printf("Usage: osu_latency [options] [RANK0 RANK1]\n\n");
        printf("RANK0 and RANK1 may be `D' or `H' which specifies whether\n"
               "the buffer is allocated on the accelerator device or host\n"
               "memory for each mpi rank\n\n");
    }

    else {
        printf("Usage: osu_latency [options]\n\n");
    }
    
    printf("options:\n");

    if (CUDA_ENABLED || OPENACC_ENABLED) {
        printf("  -d TYPE       accelerator device buffers can be of TYPE "
                "`cuda' or `openacc'\n");
    }

    printf("  -h            print this help message\n");
    fflush(stdout);
}

int
process_options (int argc, char *argv[])
{
    extern char * optarg;
    extern int optind;

    char const * optstring = (CUDA_ENABLED || OPENACC_ENABLED) ? "+d:h" : "+h";
    int c;

    /*
     * set default options
     */
    options.src = 'H';
    options.dst = 'H';

    if (CUDA_ENABLED) {
        options.accel = cuda;
    }

    else if (OPENACC_ENABLED) {
        options.accel = openacc;
    }

    else {
        options.accel = none;
    }

    while((c = getopt(argc, argv, optstring)) != -1) {
        switch (c) {
            case 'd':
                /* optarg should contain cuda or openacc */
                if (0 == strncasecmp(optarg, "cuda", 10)) {
                    if (!CUDA_ENABLED) {
                        return po_cuda_not_avail;
                    }
                    options.accel = cuda;
                }

                else if (0 == strncasecmp(optarg, "openacc", 10)) {
                    if (!OPENACC_ENABLED) {
                        return po_openacc_not_avail;
                    }
                    options.accel = openacc;
                }

                else {
                    return po_bad_usage;
                }
                break;
            case 'h':
                return po_help_message;
            default:
                return po_bad_usage;
        }
    }

    if (CUDA_ENABLED || OPENACC_ENABLED) {
        if ((optind + 2) == argc) {
            options.src = argv[optind][0];
            options.dst = argv[optind + 1][0];

            switch (options.src) {
                case 'D':
                case 'H':
                    break;
                default:
                    return po_bad_usage;
            }

            switch (options.dst) {
                case 'D':
                case 'H':
                    break;
                default:
                    return po_bad_usage;
            }
        }

        else if (optind != argc) {
            return po_bad_usage;
        }
    }

    return po_okay;
}

int
init_accel (void)
{
#if defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_)
     char * str;
     int local_rank, dev_count;
     int dev_id = 0;
#endif
#ifdef _ENABLE_CUDA_
     CUresult curesult = CUDA_SUCCESS;
     CUdevice cuDevice;
#endif

     switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                cudaGetDeviceCount(&dev_count);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            curesult = cuInit(0);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuDeviceGet(&cuDevice, dev_id);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }

            curesult = cuCtxCreate(&cuContext, 0, cuDevice);
            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                dev_count = acc_get_num_devices(acc_device_not_host); 
                fprintf(stderr, "dev_count : %d \n", dev_count);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            acc_set_device_num (dev_id, acc_device_not_host);
            break;
#endif
        default:
            fprintf(stderr, "Invalid device type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

int
allocate_device_buffer (char ** buffer)
{
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cuerr = cudaMalloc((void **)buffer, MYBUFSIZE);

            if (cudaSuccess != cuerr) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            *buffer = acc_malloc(MYBUFSIZE);
            if (NULL == *buffer) {
                fprintf(stderr, "Could not allocate device memory\n");
                return 1;
            }
            break;
#endif
        default:
            fprintf(stderr, "Could not allocate device memory\n");
            return 1;
    }

    return 0;
}

void *
align_buffer (void * ptr, unsigned long align_size)
{
    return (void *)(((unsigned long)ptr + (align_size - 1)) / align_size *
            align_size);
}

int
allocate_memory (char ** sbuf, char ** rbuf, int rank)
{
    unsigned long align_size = getpagesize();

    assert(align_size <= MAX_ALIGNMENT);

//    switch (rank) {
	if (rank == root_id) {
//        case root_id:
            if ('D' == options.src) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            }

            else {
                *sbuf = (char *)align_buffer(s_buf_original, align_size);
                *rbuf = (char *)align_buffer(r_buf_original, align_size);
            }
            //break;
	}
	else {
//        case 1:
//          default:
            if ('D' == options.dst) {
                if (allocate_device_buffer(sbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }

                if (allocate_device_buffer(rbuf)) {
                    fprintf(stderr, "Error allocating cuda memory\n");
                    return 1;
                }
            }

            else {
                *sbuf = (char *)align_buffer(s_buf_original, align_size);
                *rbuf = (char *)align_buffer(r_buf_original, align_size);
            }
//            break;
        }
//    }

    return 0;
}

void
print_header (int rank)
{
    if (root_id == rank) {
        switch (options.accel) {
            case cuda:
                printf(HEADER, "-CUDA");
                break;
            case openacc:
                printf(HEADER, "-OPENACC");
                break;
            default:
                printf(HEADER, "");
                break;
        }

        switch (options.accel) {
            case cuda:
            case openacc:
                printf("# Send Buffer on %s and Receive Buffer on %s\n",
                        'D' == options.src ? "DEVICE (D)" : "HOST (H)",
                        'D' == options.dst ? "DEVICE (D)" : "HOST (H)");
            default:
                printf("%-*s%*s\n", 10, "# Size", FIELD_WIDTH, "Latency (us)");
                fflush(stdout);
        }
    }
}

void
set_device_memory (void * ptr, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)ptr;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cudaMemset(ptr, data, size);
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
#pragma acc parallel loop deviceptr(p)
            for(i = 0; i < size; i++) {
                p[i] = data;
            }
            break;
#endif
        default:
            break;
    }
}

void
touch_data (void * sbuf, void * rbuf, int rank, size_t size)
{
    if ((root_id == rank && 'H' == options.src) ||
            (root_id != rank && 'H' == options.dst)) {
        memset(sbuf, 'a', size);
        memset(rbuf, 'b', size);
    } else {
        set_device_memory(sbuf, 'a', size);
        set_device_memory(rbuf, 'b', size);
    }
}

int
free_device_buffer (void * buf)
{
    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            cudaFree(buf);
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            acc_free(buf);
            break;
#endif
        default:
            /* unknown device */
            return 1;
    }

    return 0;
}

int
cleanup_accel (void)
{
#ifdef _ENABLE_CUDA_
     CUresult curesult = CUDA_SUCCESS;
#endif

     switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            curesult = cuCtxDestroy(cuContext);

            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            acc_shutdown(acc_device_not_host);
            break;
#endif
        default:
            fprintf(stderr, "Invalid accel type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

void
free_memory (void * sbuf, void * rbuf, int rank)
{
//    switch (rank) {
//        case root_id:
	if (rank == root_id) {
            if ('D' == options.src) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }
//            break;
//        case 1:
//         default:
	}
	else {
            if ('D' == options.dst) {
                free_device_buffer(sbuf);
                free_device_buffer(rbuf);
            }
//            break;
	}
//    }
}

/* vi:set sw=4 sts=4 tw=80: */
