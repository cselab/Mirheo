#pragma once

#include <cstdio>
#include <cmath>
#include <cassert>

#include <unistd.h>

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	sleep(5);
	if (abort) exit(code);
    }
}

#include <mpi.h>

#define MPI_CHECK(ans) do { mpiAssert((ans), __FILE__, __LINE__); } while(0)

inline void mpiAssert(int code, const char *file, int line, bool abort=true)
{
    if (code != MPI_SUCCESS) 
    {
	char error_string[2048];
	int length_of_error_string = sizeof(error_string);
	MPI_Error_string(code, error_string, &length_of_error_string);
	 
	printf("mpiAssert: %s %s %d\n", error_string, file, line);
	 	 
	MPI_Abort(MPI_COMM_WORLD, code);
    }
}

//AoS is the currency for dpd simulations (because of the spatial locality).
//AoS - SoA conversion might be performed within the hpc kernels.
struct Particle
{
    float x[3], u[3];

    static bool initialized;
    static MPI_Datatype mytype;

    static MPI_Datatype datatype()
	{
	    if (!initialized)
	    {
		MPI_CHECK( MPI_Type_contiguous(6, MPI_FLOAT, &mytype));

		MPI_CHECK(MPI_Type_commit(&mytype));

		initialized = true;
	    }

	    return mytype;
	}
};

//why do i need this? it is there just to make the code more readable
struct Acceleration
{
    float a[3];
};

const float dt = 0.02;
const float tend = 10;
const float kBT = 0.1;
const float gammadpd = 45;
const float sigma = 3; //sqrt(2 * gammadpd * kBT);
const float sigmaf = sigma / sqrt(dt);
const float aij = 2.5; 

static const int tagbase_dpd_remote_interactions = 0;
static const int tagbase_redistribute_particles = 255;

inline void mpi_cuda_test_old(int rank, MPI_Comm comm)
{
    printf("start test\n");
    float * ptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(float) * 200));

    CUDA_CHECK(cudaMemset(ptr, rank ? 1 : 313, sizeof(float) * 200));
	
    MPI_Status status;
    if (rank == 0)
	MPI_CHECK(MPI_Send(ptr, 200, MPI_FLOAT, 1, 3999, comm));
    else if (rank == 1)
    {
	float asd[200];

	CUDA_CHECK(cudaMemcpy(asd, ptr, sizeof(float) * 200, cudaMemcpyDeviceToHost));
	printf("EARLY: reading: %f %f %f\n", asd[100], asd[101], asd[102]);
	    
	MPI_CHECK(MPI_Recv(ptr, 200, MPI_FLOAT, 0, 3999, comm, &status));


	CUDA_CHECK(cudaMemcpy(asd, ptr, sizeof(float) * 200, cudaMemcpyDeviceToHost));

	printf("reading: %f %f %f\n", asd[100], asd[101], asd[102]);
    }

    MPI_Finalize();
    printf("end test\n");
    exit(0);
}

void mpi_cuda_test(int rank, MPI_Comm comm);
void cuda_mpi_test2(MPI_Comm comm);

inline void check_isnan(float * device_ptr, const int n)
{
    float * tmp = new float[n];

    CUDA_CHECK(cudaMemcpy(tmp, device_ptr, sizeof(float) * n, cudaMemcpyDeviceToHost));

    // printf("checking now..\n");
    for(int i = 0; i < n; ++i)
	assert(!isnan(tmp[i]));
    
    delete [] tmp;
}

void check_particles(const Particle * p, const int n, const int L);
