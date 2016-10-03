// Yo ho ho ho
#define private public

#include "../core/containers.h"
#include "../core/celllist.h"
#include "../core/dpd.h"
#include "../core/halo_exchanger.h"

Logger logger;

int main(int argc, char ** argv)
{
	// Init

	int nranks, rank;
	int ranks[] = {1, 1, 1};
	int periods[] = {1, 1, 1};
	MPI_Comm cartComm;

	MPI_Init(&argc, &argv);
	logger.init(MPI_COMM_WORLD, "halo.log");

	logger.MPI_Check( MPI_Comm_size(MPI_COMM_WORLD, &nranks) );
	logger.MPI_Check( MPI_Comm_rank(MPI_COMM_WORLD, &rank) );
	logger.MPI_Check( MPI_Cart_create(MPI_COMM_WORLD, 3, ranks, periods, 0, &cartComm) );

	// Initial cells

	int3 ncells = {4, 4, 4};
	float3 domainStart = {-ncells.x / 2.0f, -ncells.y / 2.0f, -ncells.z / 2.0f};
	ParticleVector dpds(ncells, domainStart);

	const int ndens = 12;
	dpds.resize(dpds.totcells*ndens);

	srand48(0);

	printf("initializing...\n");

	int c = 0;
	for (int i=0; i<ncells.x; i++)
		for (int j=0; j<ncells.y; j++)
			for (int k=0; k<ncells.z; k++)
				for (int p=0; p<ndens * drand48(); p++)
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

	cudaStream_t defStream = 0;

	HaloExchanger halo(cartComm);
	halo.attach(&dpds);

	buildCellList((float4*)dpds.coosvels.devdata, dpds.np, dpds.domainStart, dpds.ncells, 1.0f, (float4*)dpds.pingPongBuf.devdata, dpds.cellsSize.devdata, dpds.cellsStart.devdata, defStream);

	swap(dpds.coosvels, dpds.pingPongBuf, defStream);
	logger.CUDA_Check( cudaStreamSynchronize(defStream) );
	halo.exchangeInit();
	halo.exchangeFinalize();


	// Forces
	//   || Halo
	// Integrate
	// Redistribute
	// Cell list

	return 0;
}
