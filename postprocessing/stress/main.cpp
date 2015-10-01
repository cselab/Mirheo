#include <cstdio>

#include <argument-parser.h>

int main()
{
    printf("hello\n");

    const float origin[3] = {0, 0, 0};
    const float extent[3] = {48, 48, 48};
    const bool project[3] = {true, true, true};
    const int noutputchannels = 6;

    const size_t chunksize = (1 << 29) / 9 / sizeof(float);
    
    float * const pbuf = new float[9 * chunksize];

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
    
    float * const bindata = new float[noutput];
    memset(bindata, 0, sizeof(float) * noutput);
    
    FILE * fin = fopen("../../mpi-dpd/stress/stresses-00019.data", "r");
    assert(fin);

    printf("reading...\n");
    fseek(fin, 0, SEEK_END);
    const size_t filesize = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    const size_t nparticles = filesize / 9 / sizeof(float);
    assert(filesize % (9 * sizeof(float)) == 0);

    const int nhotparticles = min(nparticles, chunksize);
    fread(pbuf, sizeof(float) * 9, nhotparticles, fin);
    
    printf("i have found %d particles\n", nparticles);
    printf("particle chunk %d\n", chunksize);

    size_t nvalid = 0;

    {
	float avgs[9];
	for(int i = 0; i < 9; ++i)
	    avgs[i] = 0;

	for(int i = 0; i < nhotparticles; ++i)
	    for(int c = 0; c < 9; ++c)
		avgs[c] += pbuf[9 * i + c];

	for(int i = 0; i < 9; ++i)
	    printf("AVG %d: %.3e\n", i, avgs[i] / nhotparticles);
    }
    
    for(int i = 0; i < nhotparticles; ++i)
    {
	int index[3];
	for(int c = 0; c < 3; ++c)
	    index[c] = (int)((pbuf[9 * i + c] - origin[c]) / binsize[c]);

	bool valid = true;
	for(int c = 0; c < 3; ++c)
	    valid &= index[c] >= 0 && index[c] < nbins[c];

	if (!valid)
	    continue;

	const int binid = index[0] + nbins[0] * (index[1] + nbins[1] * index[2]);
	++bincount[binid];
	
	const int base = noutputchannels * binid;
	
	for(int c = 0; c < 6; ++c)
	    bindata[base + c] += pbuf[9 * i + 3 + c];

	++nvalid;
    }

    const bool avg = true;
    if (avg)
	for(int i = 0; i < ntotbins; ++i)
	{
	    printf("bincount %03d\n", bincount[i]);

	    for(int c = 0; c < 6; ++c)
		bindata[9 * i + c] /= bincount[i];
	}


    int nprojections = 0;
    for(int c = 0; c < 3; ++c)
	nprojections += project[c];

    if (nprojections == 3) //six numbers
    {
	assert(noutput == noutputchannels);
	printf("result: ");

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
			printf("%+.3e ", bindata[noutputchannels * ctr + c]);

		    printf("\n");

		    ++ctr;
		}
    }
    else if (nprojections == 1)
    {	
	int nx;
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
			printf("%+.3e ", bindata[noutputchannels * ctr + c]);

			++ctr;

			if (ctr % nx == 0)
			    printf("\n");
		    }
	}
    }
    
    printf("valid: %d\n", nvalid);
    
    fclose(fin);

    delete [] pbuf;
    delete [] bincount;
    delete [] bindata;

    return 0;
}
