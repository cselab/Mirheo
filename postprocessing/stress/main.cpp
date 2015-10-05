#include <mpi.h>
#include <unistd.h>
#include <fcntl.h>

#include <cstdio>
#include <iostream>
#include <vector>
#include <sstream>

#include <argument-parser.h>
#include <mpi-check.h>

using namespace std;

int main(int argc, const char ** argv)
{
    MPI_CHECK( MPI_Init(&argc, (char ***)&argv) );

    int nranks, rank;
    MPI_CHECK( MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK( MPI_Comm_size(MPI_COMM_WORLD, &nranks));

    ArgumentParser argp(argc, argv);

    const bool verbose = argp("-verbose").asBool(false);
    const bool avg = argp("-average").asBool(true);
    vector<float> origin = argp("-origin").asVecFloat(3);
    vector<float> extent = argp("-extent").asVecFloat(3);
    vector<float> projectf = argp("-project").asVecFloat(3);
    string contributions = argp("-contributions").asString("uf");

    const double ufactor = contributions.find("u") != string::npos;
    const double ffactor = contributions.find("f") != string::npos;

    bool project[3];
    for(int c = 0; c < 3; ++c)
	project[c] = projectf[c] != 0;

    int nprojections = 0;
    for(int c = 0; c < 3; ++c)
	nprojections += project[c];

    const int noutputchannels = 6;
    const size_t chunksize = (1 << 29) / 12 / sizeof(float);

    float * const pbuf = new float[12 * chunksize];

    float binsize[3];
    for(int c = 0; c < 3; ++c)
	binsize[c] = project[c] ? extent[c] : 1;

    int nbins[3];
    for(int c = 0; c < 3; ++c)
	nbins[c] = extent[c] / binsize[c];

    const int ntotbins = nbins[0] * nbins[1] * nbins[2];

    int * const bincount = new int[ntotbins];
    memset(bincount, 0, sizeof(int) * ntotbins);

    const int noutput = noutputchannels * ntotbins;

    double * const bindata = new double[noutput];
    memset(bindata, 0, sizeof(double) * noutput);

    vector<string> paths;

    {
	string myinput;

	if (rank == 0)
	    for (string line; getline(cin, line);)
		myinput += line + "\n";

	int inputsize = myinput.size();
	MPI_CHECK( MPI_Bcast(&inputsize, 1, MPI_INTEGER, 0, MPI_COMM_WORLD));

	myinput.resize(inputsize);

	MPI_CHECK( MPI_Bcast(&myinput[0], inputsize, MPI_CHAR, 0, MPI_COMM_WORLD));

	int c = 0;
	istringstream iss(myinput);

	for (string line; getline(iss, line); ++c)
	    if (c % nranks == rank)
		paths.push_back(line);
    }

    int numfiles = paths.size();

    size_t totalfootprint = 0;
    double timeIO = 0;

    for(int ipath = 0; ipath < (int)paths.size(); ++ipath)
    {
	const char * const path = paths[ipath].c_str();

	if (verbose)
	    fprintf(stderr, "working on <%s>\n", path);

	int fdin = open(path, O_RDONLY);

	if (!fdin)
	{
	    fprintf(stderr, "can't access <%s> , exiting now.\n", path);
	    exit(-1);
	}

	if (verbose)
	    perror("reading...\n");

	const size_t filesize = lseek(fdin, 0, SEEK_END);

	totalfootprint += filesize;

	lseek(fdin, 0, SEEK_SET);

	const size_t nparticles = filesize / 12 / sizeof(float);
	assert(filesize % (12 * sizeof(float)) == 0);

	if (verbose)
	{
	    fprintf(stderr, "i have found %d particles\n", (int)nparticles);
	    fprintf(stderr, "particle chunk %d\n", (int)chunksize);
	}

	for(size_t base = 0; base < nparticles; base += chunksize)
	{
	    const int nhotparticles = min(nparticles - base, chunksize);
	    const size_t nhotbytes = nhotparticles * sizeof(float) * 12;

	    size_t nreadbytes = 0;
	    int start = 0;

	    while(start < nhotparticles)
	    {
		const double tstart = MPI_Wtime();
		nreadbytes += read(fdin, pbuf, nhotbytes - nreadbytes);
		timeIO += MPI_Wtime() - tstart;

		const int stop = nreadbytes / sizeof(float) / 12;

#ifndef NDEBUG
		if (verbose)
		{
		    float avgs[12];
		    for(int i = 0; i < 12; ++i)
			avgs[i] = 0;

		    for(int i = 0; i < nhotparticles; ++i)
			for(int c = 0; c < 12; ++c)
			    avgs[c] += pbuf[12 * i + c];

		    for(int i = 0; i < 12; ++i)
			printf("AVG %d: %.3e\n", i, avgs[i] / nhotparticles);
		}
#endif

		for(int i = start; i < stop; ++i)
		{
		    const int srcbase = 12 * i;

		    int index[3];
		    for(int c = 0; c < 3; ++c)
			index[c] = (int)((pbuf[srcbase + c] - origin[c]) / binsize[c]);

		    bool valid = true;
		    for(int c = 0; c < 3; ++c)
			valid &= index[c] >= 0 && index[c] < nbins[c];

		    if (!valid)
			continue;

		    const int binid = index[0] + nbins[0] * (index[1] + nbins[1] * index[2]);
		    ++bincount[binid];

		    const int dstbase = noutputchannels * binid;

		    const int v1[6] = {0, 0, 0, 1, 1, 2};
		    const int v2[6] = {0, 1, 2, 1, 2, 2};

		    for(int c = 0; c < 6; ++c)
			bindata[dstbase + c] +=
			    ffactor * pbuf[srcbase + 6 + c] +
			    ufactor * pbuf[srcbase + 3 + v1[c]] * pbuf[srcbase + 3 + v2[c]];
		}

		start = stop;
	    }
	}

	close(fdin);
    }

    if (!numfiles)
    {
	perror("ooops zero files were read. Exiting now.\n");
	exit(-1);
    }

    MPI_CHECK( MPI_Reduce(rank ? bincount : MPI_IN_PLACE, bincount, ntotbins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) );
    MPI_CHECK( MPI_Reduce(rank ? bindata : MPI_IN_PLACE, bindata, noutput, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
    MPI_CHECK( MPI_Reduce(rank ? &timeIO : MPI_IN_PLACE, &timeIO, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD) );
    MPI_CHECK( MPI_Reduce(rank ? &totalfootprint : MPI_IN_PLACE, &totalfootprint, 1, MPI_OFFSET, MPI_SUM, 0, MPI_COMM_WORLD) );

    if (rank)
	goto finalize;

    if (avg)
	for(int i = 0; i < ntotbins; ++i)
	    for(int c = 0; c < noutputchannels; ++c)
		bindata[noutputchannels * i + c] /= bincount[i];

    if (nprojections == 3)
    {
	assert(noutput == noutputchannels);

	for(int c = 0; c < noutputchannels; ++c)
	    printf("%+.3e\t", bindata[c]);

	printf("\n");
    }
    else if (nprojections == 2)
    {
	int ctr = 0;
	for(int iz = 0; iz < nbins[2]; ++iz)
	    for(int iy = 0; iy < nbins[1]; ++iy)
		for(int ix = 0; ix < nbins[0]; ++ix)
		{
		    printf("%03d ", ctr);

		    for(int c = 0; c < noutputchannels; ++c)
			printf("%+.4e ", bindata[noutputchannels * ctr + c]);

		    printf("\n");

		    ++ctr;
		}
    }
    else if (nprojections == 1)
    {
	int nx = 0;
	for(int c = 0; c < 3; ++c)
	    if (nbins[c] > 1)
	    {
		nx = nbins[c];
		break;
	    }

	for(int c = 0; c < noutputchannels; ++c)
	{
	    int ctr = 0;

	    for(int iz = 0; iz < nbins[2]; ++iz)
		for(int iy = 0; iy < nbins[1]; ++iy)
		    for(int ix = 0; ix < nbins[0]; ++ix)
		    {
			printf("%+.5e ", bindata[noutputchannels * ctr + c]);

			++ctr;

			if (ctr % nx == 0)
			    printf("\n");
		    }

	    if (c < noutputchannels - 1)
		printf("\n");
	}
    }
    else
    {
	perror("woops invalid number of projections. Exiting now...\n");
	exit(-1);
    }

    if (verbose)
	perror("all is done. ciao.\n");

    fprintf(stderr, "total footprint: %.3f MB, I/O time: %.3f ms\n", totalfootprint * 1. / 1024 / 1024, timeIO * 1e3);
    fprintf(stderr, "read throughput: %.3f GB/s\n", totalfootprint /( 1024 * 1024) / timeIO / 1024);

finalize:

    delete [] pbuf;
    delete [] bincount;
    delete [] bindata;

    MPI_CHECK( MPI_Finalize() );

    return 0;
}
