#include <core/datatypes.h>
#include <core/scan.h>

#include <cstdio>
#include <cstdint>

int main()
{
	const int n = 96*96*96;

	PinnedBuffer<uint8_t> inp(n);
	PinnedBuffer<uint> out(n);
	HostBuffer<uint> refout(n+1);

	for (int i=0; i<n; i++)
		inp[i] = (uint8_t)(drand48() * 250);

	inp.synchronize(synchronizeDevice);
	for (int i=0; i<50; i++)
		scan(inp.devdata, n, out.devdata, 0);
	out.synchronize(synchronizeHost);

	refout[0] = 0;
	for (int i=1; i<n; i++)
	{
		refout[i] = refout[i-1] + inp[i-1];

		if (refout[i] != out[i]) printf("%4d   %3d:  %5d,  %5d\n", i, inp[i], refout[i], out[i]);
	}

	return 0;
}
