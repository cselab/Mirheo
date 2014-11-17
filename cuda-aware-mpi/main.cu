#include <cstdio>
#include <cmath>

#include <mpi.h>

#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	
	if (abort)
	    exit(code);
    }
}


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

void cuda_awareness_test(MPI_Comm comm)
{
    const int shift = 0;
    const int nmessages = 1;
    const bool cuda = true;
    
    int myrank;
    MPI_CHECK( MPI_Comm_rank(comm, &myrank) );

    int nranks;
    MPI_CHECK( MPI_Comm_size(comm, &nranks) );
    
    printf("hello test %d\n", myrank);

   
    float * bufsend;

    if (cuda)
	CUDA_CHECK(cudaMalloc(&bufsend, sizeof(float) * nmessages));
    else
	bufsend = new float[nmessages];
    
    printf("sending (%d)..\n", myrank);
    
    MPI_Request sendreq[nmessages];
    for(int i = 0; i < nmessages; ++i)
    	MPI_CHECK( MPI_Isend(bufsend + i, 1, MPI_FLOAT, (myrank + nranks + shift) % nranks, i, comm, sendreq + i));

    float * bufrecv;
    if (cuda)
	CUDA_CHECK(cudaMalloc(&bufrecv, sizeof(float) * nmessages));
    else
	bufrecv = new float[nmessages];
    
    printf("receiving (%d)...\n", myrank);
    
    MPI_Request recvreq[nmessages];
    
    for(int i = 0; i < nmessages; ++i)
	MPI_CHECK( MPI_Irecv(bufrecv + i, 1, 
			     MPI_FLOAT, (myrank + nranks - shift) % nranks, i, comm, recvreq + i) );
    
    MPI_Status statuses[nmessages];
    MPI_CHECK( MPI_Waitall(nmessages, recvreq, statuses) );
    MPI_CHECK( MPI_Waitall(nmessages, sendreq, statuses) );

    if (cuda)
    {
	CUDA_CHECK(cudaFree(bufrecv));
	CUDA_CHECK(cudaFree(bufsend));
    }
    else
    {
	delete [] bufrecv;
	delete [] bufsend;
    }

    printf("end test\n");
}

__global__ void myallocation(float ** ptr)
{
    *ptr = new float[1];
    printf("allocated ptr %p\n", *ptr);
}

void inspect(const void * ptr)
{
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, ptr));
    char * memtype = "unknown";

    if (attributes.memoryType == cudaMemoryTypeHost)
	memtype = "HOSTMEM";

    if (attributes.memoryType == cudaMemoryTypeDevice)
	memtype = "DEVICE";

    
    printf("INSPECT: %p: Type: %s pointer, devPtr: %p, hostPtr: %p\n", ptr, memtype, attributes.devicePointer, attributes.hostPointer);
}

void cuda_awareness_test2()
{
    float * bufsend;

    //allocate bufsend
    {
	float ** ptr;
	
	CUDA_CHECK(cudaMalloc(&ptr, sizeof(float *)));
	myallocation<<<1, 1>>>(ptr);
	CUDA_CHECK(cudaMemcpy(&bufsend, ptr, sizeof(float *), cudaMemcpyDeviceToHost));
	printf("my SEND ptr is %p\n", bufsend);
    }

    float * bufrecv;

    //allocate bufrecv
    {
	float ** ptr;
		
	CUDA_CHECK(cudaMalloc(&ptr, sizeof(float *)));
	myallocation<<<1, 1>>>(ptr);
	CUDA_CHECK(cudaMemcpy(&bufrecv, ptr, sizeof(float *), cudaMemcpyDeviceToHost));
	printf("my RECV ptr is %p\n", bufrecv);
    }
   
    inspect(bufsend);
    inspect(bufrecv);
    
    CUDA_CHECK(cudaMemcpy(bufrecv, bufsend, sizeof(float), cudaMemcpyDeviceToDevice));
    
    printf("end test\n");
}

int main(int argc, char ** argv)
{
    MPI_CHECK( MPI_Init(&argc, &argv) );

    cuda_awareness_test(MPI_COMM_WORLD);
    //cuda_awareness_test2();
    
    MPI_Finalize();
}