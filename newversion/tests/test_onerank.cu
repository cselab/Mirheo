// Yo ho ho ho
#define private public

#include "../core/containers.h"
#include "../core/celllist.h"
#include "../core/dpd.h"
#include "../core/halo_exchanger.h"
#include "../core/logger.h"
#include "../core/integrate.h"

#include <unistd.h>

Logger logger;

int main(int argc, char ** argv)
{
	// Init

	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE)
	{
	    printf("ERROR: The MPI library does not have full thread support\n");
	    MPI_Abort(MPI_COMM_WORLD, 1);
	}

	logger.init(MPI_COMM_WORLD, "onerank.log", 9);

	MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	// Initial cells

	int3 ncells = {64, 64, 64};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	float3 length{(float)ncells.x, (float)ncells.y, (float)ncells.z};
	ParticleVector dpds(ncells, domainStart, length);

	const int ndens = 8;
	dpds.resize(ncells.x*ncells.y*ncells.z * ndens);

	srand48(0);

	printf("initializing...\n");

	int c = 0;
	for (int i=0; i<ncells.x; i++)
		for (int j=0; j<ncells.y; j++)
			for (int k=0; k<ncells.z; k++)
				for (int p=0; p<ndens; p++)
				{
					dpds.coosvels[c].x[0] = i + drand48() + domainStart.x;
					dpds.coosvels[c].x[1] = j + drand48() + domainStart.y;
					dpds.coosvels[c].x[2] = k + drand48() + domainStart.z;
					dpds.coosvels[c].i1 = c;

					dpds.coosvels[c].u[0] = drand48() - 0.5;
					dpds.coosvels[c].u[1] = drand48() - 0.5;
					dpds.coosvels[c].u[2] = drand48() - 0.5;
					c++;
				}

	dpds.resize(c);
	dpds.coosvels.synchronize(synchronizeDevice);

	cudaStream_t defStream;
	CUDA_Check( cudaStreamCreateWithPriority(&defStream, cudaStreamNonBlocking, 10) );

	HaloExchanger halo(cartComm);
	halo.attach(&dpds, 7);

	buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, dpds.totcells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);
	swap(dpds.coosvels, dpds.pingPongBuf, defStream);
	CUDA_Check( cudaStreamSynchronize(defStream) );

	for (int i=0; i<100; i++)
	{
		buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, dpds.totcells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);
		swap(dpds.coosvels, dpds.pingPongBuf, defStream);
		cudaStreamSynchronize(defStream);

		computeInternalDPD(dpds, defStream);

		halo.exchangeInit();
		halo.exchangeFinalize();

		computeHaloDPD(dpds, defStream);
		integrate(dpds, 1e-15f, defStream);


		//cudaDeviceSynchronize();



		//cudaDeviceSynchronize();

		cudaStreamSynchronize(defStream);
	}




	return 0;
}
