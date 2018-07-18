#include <core/containers.h>
#include <core/datatypes.h>
#include <core/scan.h>

#include <core/cub/device/device_scan.cuh>

#include <cstdio>
#include <cstdint>

Logger logger;

int main()
{
	const int n = 64*64*64*8;

	PinnedBuffer<uint8_t> inp(n);
	PinnedBuffer<int> out(n);
	HostBuffer<uint> refout(n+1);

	DeviceBuffer<char> buf;

	for (int i=0; i<n; i++)
		inp[i] = (uint8_t)(drand48() * 250);

	inp.uploadToDevice(0);
	for (int i=0; i<100; i++)
	{
		scan(inp.devPtr(), n, out.devPtr(), 0);

		size_t bufSize;
		cub::DeviceScan::ExclusiveSum(nullptr, bufSize, inp.devPtr(), out.devPtr(), n);
		// Allocate temporary storage
		buf.resize(bufSize, 0);
		// Run exclusive prefix sum
		cub::DeviceScan::ExclusiveSum(buf.devPtr(), bufSize, inp.devPtr(), out.devPtr(), n);
	}

	out.downloadFromDevice(0);

	int c = 0;
	refout[0] = 0;
	for (int i=1; i<n; i++)
	{
		refout[i] = refout[i-1] + inp[i-1];

		if (refout[i] != out[i])
		{
			printf("%4d   %3d:  %5d,  %5d\n", i, inp[i], refout[i], out[i]);
			if (c++ > 10)
				return 0;
		}
	}

	return 0;
}
