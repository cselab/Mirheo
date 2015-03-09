#include <vector>
#include <mpi.h>
#include "common.h"
#include "containers.h"

__global__ void simpleSamplingKernel(const float *xyzuvw, float *bins, int *binSize, int nRbins, int nPhibins, int npart,
		float shY, float shZ, float Rpipe);

class QOS
{
	int nranks[3];
	int coords[3];
	int myrank;
	MPI_Comm cartcomm;

	FILE *fout, *fvisc, *fdens;

	PinnedHostBuffer<float> bins;
	PinnedHostBuffer<int>   binSize;

	float *allbins, *avgByPhibins, *allvisc;
	int   *allBinSize;

	const int nRbins, nPhibins;

	float n, force;
	int nsamples;

public:
	QOS(int nranks[3], int coords[3], int myrank, MPI_Comm cartcomm, int nRbins, int nPhibins, float n, float force);
	void exec(float tm, const Particle* ctc_xyzuvw, int npart, bool dump);
	~QOS();
};

