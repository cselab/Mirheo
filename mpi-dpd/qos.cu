#include "qos.h"
#include <algorithm>

using namespace std;

__global__ void simpleSamplingKernel(const float *xyzuvw, float *bins, int *binSize, int nRbins, int nPhibins, int npart,
        float shY, float shZ, float Rpipe)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= npart) return;

    const float y = xyzuvw[6*pid + 1] + shY;
    const float z = xyzuvw[6*pid + 2] + shZ;
    const float r = sqrt(y*y + z*z);
    const float phi = atan2(z, y) + M_PI;

    int rbin = r / Rpipe * nRbins;
    if (rbin < 0) rbin = 0;
    if (rbin >= nRbins) rbin = nRbins - 1;

    const int phibin = phi / (2*M_PI) * nPhibins;
    const int ibin = phibin*nRbins + rbin;

    atomicAdd(binSize+ibin, 1);
    atomicAdd(bins+ibin, (xyzuvw[6*pid + 3]));
}

QOS::QOS(int nranks[3], int coords[3], int myrank, MPI_Comm cartcomm, int nRbins, int nPhibins, float n, float force) :
								myrank(myrank), cartcomm(cartcomm), nRbins(nRbins), nPhibins(nPhibins), n(n), force(force)
{
	for (int i=0; i<3; i++)
	{
		this->nranks[i] = nranks[i];
		this->coords[i] = coords[i];
	}

	if (myrank == 0)
	{
		fout  = fopen("velprof.dat", "w");
		fvisc = fopen("viscosity.dat", "w");
		fdens = fopen("density.dat", "w");
	}

	bins.resize(nRbins * nPhibins);
	binSize.resize(nRbins * nPhibins);
	nsamples = 0;

	cudaMemset(bins.devptr,    0, nRbins*nPhibins*sizeof(float));
	cudaMemset(binSize.devptr, 0, nRbins*nPhibins*sizeof(int));

	if (myrank == 0)
	{
		allbins = new float[nRbins * nPhibins];
		avgByPhibins = new float[nRbins];
		allBinSize = new int[nRbins * nPhibins];
		allvisc    = new float[nPhibins];
	}
}

QOS::~QOS()
{
	if (myrank == 0)
	{
		fclose(fout);
		fclose(fvisc);
		fclose(fdens);
	}
}

inline float vel(float r, float Rpipe, float vmax)
{
	return vmax * (1.0f - pow(r/Rpipe, 2));
}

void QOS::exec(float tm, const Particle* ctc_xyzuvw, int npart, bool dump)
{
	const float shY = coords[1] * YSIZE_SUBDOMAIN + 0.5*YSIZE_SUBDOMAIN - nranks[1]*YSIZE_SUBDOMAIN*0.5;
	const float shZ = coords[2] * ZSIZE_SUBDOMAIN + 0.5*ZSIZE_SUBDOMAIN - nranks[2]*ZSIZE_SUBDOMAIN*0.5;
	const float Rpipe = nranks[1]*YSIZE_SUBDOMAIN*0.5 * 40.0/48.0;

	nsamples++;

	simpleSamplingKernel<<<(127 + npart)/128, 128>>>
			(&ctc_xyzuvw[0].x[0], bins.devptr, binSize.devptr, nRbins, nPhibins, npart, shY, shZ, Rpipe);

	if (dump)
	{
		MPI_Reduce(binSize.data,  allBinSize, nRbins*nPhibins, MPI_INT,   MPI_SUM, 0, cartcomm);
		MPI_Reduce(bins.data,     allbins,    nRbins*nPhibins, MPI_FLOAT, MPI_SUM, 0, cartcomm);

		cudaMemset(bins.devptr,    0, nRbins*nPhibins*sizeof(float));
		cudaMemset(binSize.devptr, 0, nRbins*nPhibins*sizeof(int));

		if (myrank == 0)
		{
			for (int i=0; i<nRbins*nPhibins; i++)
				if (allBinSize[i] != 0)
					allbins[i] /= allBinSize[i];

			vector<float> binsvec(allbins, allbins + nRbins*nPhibins);

			float avgvmax = 0;
			for (int j=0; j<nPhibins; j++)
			{
				float vmax = *(max_element(binsvec.begin() + j*nRbins, binsvec.begin() + (j+1)*nRbins));
				allvisc[j] = n * force * Rpipe * Rpipe / 4.0f / vmax;
				avgvmax += vmax;
			}
			avgvmax /= nPhibins;

			for (int i=0; i<nRbins; i++)
			{
				avgByPhibins[i] = 0;
				for (int j=0; j<nPhibins; j++)
					avgByPhibins[i] += allbins[j*nRbins + i];

				avgByPhibins[i] /= nPhibins;
			}

			fprintf(fout, "\n%f\n", tm);
			for (int i=0; i<nRbins; i++)
			{
				fprintf(fout, "%d  %e   %e    ", i, (i+0.5) * Rpipe / nRbins, avgByPhibins[i]);
				for (int j=0; j < nPhibins; j++)
					fprintf(fout, "  %e", allbins[j*nRbins + i]);
				fprintf(fout, "\n");
			}
			fflush(fout);

			// density
			fprintf(fdens, "\n%f\n", tm);
			for (int i=0; i<nRbins; i++)
			{
				float rmin = i*Rpipe / nRbins;
				float rmax = (i+1)*Rpipe / nRbins;
				float vol = M_PI * (rmax*rmax - rmin*rmin) * XSIZE_SUBDOMAIN * nranks[0];

				int sum = 0;
				for (int j=0; j<nPhibins; j++)
					sum += allBinSize[j*nRbins + i];

				fprintf(fdens, "%d  %e   %e\n", i, (i+0.5) * Rpipe / nRbins, sum / vol / nsamples);
			}
			fflush(fdens);


			float avgvisc = 0;
			for (int i=0; i<nPhibins; i++)
				avgvisc += allvisc[i];
			avgvisc /= nPhibins;

			float l2norm2 = 0.0f;
			for (int i = 0; i < nRbins; ++i)
			{
				float estVel = vel(i * Rpipe / nRbins, Rpipe, avgvmax);
				l2norm2 += pow(avgByPhibins[i] - estVel, 2);
			}

			float l2norm = sqrt(l2norm2);


			fprintf(fvisc, "\n%f\n%e  %e   ", tm, avgvisc, l2norm);
			for (int i=0; i<nPhibins; i++)
				fprintf(fvisc, "    %e", allvisc[i]);
			fprintf(fvisc, "\n");
			fflush(fvisc);
		}

		nsamples = 0;
	}
}
