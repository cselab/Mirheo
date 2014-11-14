#include "common.h"

bool Particle::initialized = false;

MPI_Datatype Particle::mytype;

void mpi_cuda_test(int rank, MPI_Comm comm)
{
    printf("start test\n");
    
    const int nentries = 20;
        
    int * ptr;
    CUDA_CHECK(cudaMalloc(&ptr, sizeof(int) * nentries));

    int val =  rank ? 1 : 2;
    
    CUDA_CHECK(cudaMemset(ptr, val, sizeof(int) * nentries));

    int * asd = new int[nentries];
    
    CUDA_CHECK(cudaMemcpy(asd, ptr, sizeof(int) * nentries, cudaMemcpyDeviceToHost));
    
     printf("INITIAL (rank %d): ", rank);
	for(int i = 0; i < nentries; ++i)
	    printf("%d ", asd[i] & 0xff);
	printf("\n");
    
    MPI_Status status;
    
    if (rank == 0)
	MPI_CHECK(MPI_Send(ptr, nentries, MPI_INT, 1, 3999, comm));
    
    if (rank == 1)
    {
	CUDA_CHECK(cudaMemcpy(asd, ptr, sizeof(int) * nentries, cudaMemcpyDeviceToHost));

	printf("BEFORE: ");
	for(int i = 0; i < nentries; ++i)
	    printf("%d ", asd[i] & 0xff);
	printf("\n");
	
	MPI_CHECK(MPI_Recv(ptr, nentries, MPI_INT, 0, 3999, comm, &status));

	CUDA_CHECK(cudaMemcpy(asd, ptr, sizeof(int) * nentries, cudaMemcpyDeviceToHost));

	printf("AFTER: ");
	for(int i = 0; i < nentries; ++i)
	    printf("%d ", asd[i] & 0xff);
	printf("\n");
    }

    MPI_Finalize();
    printf("end test\n");
    exit(0);
}

void cuda_mpi_test2_particle(MPI_Comm comm)
{
    int myrank;
    MPI_CHECK( MPI_Comm_rank(comm, &myrank) );

    int nranks;
    MPI_CHECK( MPI_Comm_size(comm, &nranks) );
    
    printf("hello test %d\n", myrank);
    
    Particle * bufsend;
    bufsend = new Particle[26];
    //CUDA_CHECK(cudaMalloc(&bufsend, sizeof(Particle) * 26));

    const int shift = 1;
    
    printf("sending..\n");
    MPI_Request sendreq[26];
    for(int i = 0; i < 26; ++i)
    	MPI_CHECK( MPI_Isend(bufsend + i, 1, Particle::datatype(), (myrank + nranks + shift) % nranks, i, comm, sendreq + i));

    Particle * bufrecv;
    bufrecv = new Particle[26];
    //CUDA_CHECK(cudaMalloc(&bufrecv, sizeof(Particle) * 26));

    printf("receiving...\n");
    MPI_Request recvreq[26];
    
    for(int i = 0; i < 26; ++i)
	MPI_CHECK( MPI_Irecv(bufrecv + i, 1, 
			     Particle::datatype(),  (myrank + nranks - shift) % nranks, i, comm, recvreq + i) );

    MPI_Status statuses[26];
    MPI_CHECK( MPI_Waitall(26, recvreq, statuses) );
    MPI_CHECK( MPI_Waitall(26, sendreq, statuses) );

    MPI_Finalize();

    printf("end test\n");
    exit(0);
}

void cuda_mpi_test2(MPI_Comm comm)
{
    const int shift = 1;
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
    
    printf("sending..\n");
    MPI_Request sendreq[nmessages];
    for(int i = 0; i < nmessages; ++i)
    	MPI_CHECK( MPI_Isend(bufsend + i, 1, MPI_FLOAT, (myrank + nranks + shift) % nranks, i, comm, sendreq + i));

    float * bufrecv;

    if (cuda)
	CUDA_CHECK(cudaMalloc(&bufrecv, sizeof(float) * nmessages));
    else
	bufrecv = new float[nmessages];
    
    printf("receiving...\n");
    MPI_Request recvreq[nmessages];
    
    for(int i = 0; i < nmessages; ++i)
	MPI_CHECK( MPI_Irecv(bufrecv + i, 1, 
			     MPI_FLOAT, (myrank + nranks - shift) % nranks, i, comm, recvreq + i) );

    MPI_Status statuses[nmessages];
    MPI_CHECK( MPI_Waitall(nmessages, recvreq, statuses) );
    MPI_CHECK( MPI_Waitall(nmessages, sendreq, statuses) );

    MPI_Finalize();

    printf("end test\n");

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
    
    exit(0);
}

__global__ void _check_particles_kernel(const Particle * p, const int n, const int L)
{
    assert(blockDim.x * gridDim.x >= n);

    const int pid = threadIdx.x + blockDim.x * blockIdx.x;

    if (pid >= n)
	return;

    for(int c = 0; c < 3; ++c)
	assert(p[pid].x[c] >= -L / 2 && p[pid].x[c] < L / 2);
}


void check_particles(const Particle * p, const int n, const int L)
{
    if (n > 0)
	_check_particles_kernel<<< (n + 127) / 128, 128 >>>(p, n, L);
    
    CUDA_CHECK(cudaPeekAtLastError());
}