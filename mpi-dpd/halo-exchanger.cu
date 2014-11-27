#include <cstring>
#include <algorithm>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>

#include "halo-exchanger.h"

using namespace std;

HaloExchanger::HaloExchanger(MPI_Comm _cartcomm, int L):  L(L)
{
    assert(L % 2 == 0);
    assert(L >= 2);

    MPI_CHECK( MPI_Comm_dup(_cartcomm, &cartcomm));

    MPI_CHECK( MPI_Comm_rank(cartcomm, &myrank));
    MPI_CHECK( MPI_Comm_size(cartcomm, &nranks));

    MPI_CHECK( MPI_Cart_get(cartcomm, 3, dims, periods, coords) );

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

	recv_tags[i] = tagbase_dpd_remote_interactions + (2 - d[0]) % 3 + 3 * ((2 - d[1]) % 3 + 3 * ((2 - d[2]) % 3));

	int coordsneighbor[3];
	for(int c = 0; c < 3; ++c)
	    coordsneighbor[c] = coords[c] + d[c];

	MPI_CHECK( MPI_Cart_rank(cartcomm, coordsneighbor, dstranks + i) );

	const int nhalocells = pow(L, 3 - fabs(d[0]) - fabs(d[1]) - fabs(d[2]));
	const int estimate = 2 * 3 * nhalocells;
	recvbufs[i].resize(estimate);
	sendbufs[i].resize(estimate);
	scattered_entries[i].resize(estimate);
	
	sendcellstarts[i].resize(nhalocells + 1);
	tmpcount[i].resize(nhalocells + 1);
	tmpstart[i].resize(nhalocells + 1);
    }

    CUDA_CHECK(cudaHostAlloc((void **)&required_send_bag_size, sizeof(int) * 26, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&required_send_bag_size_host, required_send_bag_size, 0));
}

namespace PackingHalo
{
    __device__ int blockcount, global_histo[27], requiredsize;

    __global__ void setup()
    {
	blockcount = 0;
	requiredsize = 0;

	for(int i = 0; i < 27; ++i)
	    global_histo[i] = 0;
    }

    template< int work >
    __global__ void count(int * const packs_start, const Particle * const p, const int np, const int L, 
			  int * bag_size_required)
    {
	assert(blockDim.x * gridDim.x * work >= np);
	assert(blockDim.x >= 26);

	__shared__ int histo[26];

	const int tid = threadIdx.x; 

	if (tid < 26)
	    histo[tid] = 0;

	__syncthreads();

	for(int t = 0; t < work; ++t)
	{
	    const int pid = tid + blockDim.x * (blockIdx.x + gridDim.x * t);

	    if (pid < np)
		for(int i = 0; i < 26; ++i)
		{
		    int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		    bool halo = true;			

		    for(int c = 0; c < 3; ++c)
		    {
			const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
			const float halo_end = min(d[c] * L + L/2 + 1, L/2);

			const float x = p[pid].x[c];

			halo &= (x >= halo_start && x < halo_end);
		    }

		    if (halo)
			atomicAdd(histo + i, 1);
		}
	}

	__syncthreads();

	if (tid < 26 && histo[tid] > 0)
	    atomicAdd(global_histo + tid, histo[tid]);

	if (tid == 0)
	{
	    const int timestamp = atomicAdd(&blockcount, 1);

	    if (timestamp == gridDim.x - 1)
	    {
		blockcount = 0;

		int s = 0, curr;

		for(int i = 0; i < 26; ++i)
		{
		    curr = global_histo[i];
		    global_histo[i] = packs_start[i] = s;
		    s += curr;
		}

		global_histo[26] = packs_start[26] = s;
		requiredsize = s;		
		*bag_size_required = s;
	    }
	}
    }

    __global__ void count(const int * const cellsstart, const int * const cellscount,
			  const int3 halo_offset, const int3 halo_size, const int L,
			  int * const output_start, int * const output_count)
    {
	assert(halo_size.x * halo_size.y * halo_size.z <= blockDim.x * gridDim.x);

	const int gid = threadIdx.x + blockDim.x * blockIdx.x;

	const int3 tmp = make_int3(gid % halo_size.x, (gid / halo_size.x) % halo_size.y, (gid / (halo_size.x * halo_size.y)));
	 
	const int3 dst = make_int3(halo_offset.x + tmp.x,
				   halo_offset.y + tmp.y,
				   halo_offset.z + tmp.z);

	const int nsize = halo_size.x * halo_size.y * halo_size.z;
	 
	if (gid < nsize)
	{
	    assert(dst.x >= 0 && dst.x < L);
	    assert(dst.y >= 0 && dst.y < L);
	    assert(dst.z >= 0 && dst.z < L);
	    
	    const int srcentry = dst.x + L * (dst.y + L * dst.z);

	    assert(srcentry < L * L * L);

	    output_start[gid] = cellsstart[srcentry];
	    output_count[gid] = cellscount[srcentry];
	}
	else
	    if (gid == nsize)
		output_start[gid] = output_count[gid] = 0;
    }

    __global__ void fill(const Particle * const particles, const int np,
			 const int * const start_src, const int * const count_src, const int L,
			 const int * const start_dst, 
			 Particle * const bag, const int bagsize, int * const scattered_entries, int * const required_bag_size, const int code)
    {
	assert(blockDim.x == warpSize);

	const int cellid = blockIdx.x;
	const int tid = threadIdx.x;

	const int base_src = start_src[cellid];
	const int base_dst = start_dst[cellid];

	const int nsrc = min(count_src[cellid], bagsize - base_dst);
	
	for(int i = tid; i < nsrc; i += warpSize)
	{
	    const int pid = base_src + i;

	    assert(pid < np && pid >= 0);
	    
	    bag[base_dst + i] = particles[pid];
	    scattered_entries[base_dst + i] = pid;

#ifndef NDEBUG
	    {
		int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };
		
		for(int c = 0; c < 3; ++c)
		{
		    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
		    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

		    if (!(particles[pid].x[c] >= halo_start && particles[pid].x[c] < halo_end))
		    {
			printf("oooops particle %d: %e %e %e component %d not within %f , %f\n", pid, particles[pid].x[0], particles[pid].x[1], particles[pid].x[2],
			       c, halo_start, halo_end);
		
		    }
		    const float eps = 1e-5;
		    assert(particles[pid].x[c] >= halo_start - eps && particles[pid].x[c] < halo_end + eps);
		}
	    }
#endif
	}

	if (cellid == gridDim.x - 1)
	    *required_bag_size = base_dst;
    }
     
    __global__ void pack(const Particle * const particles, const int np, const int L, Particle * const bag, 
			 const int bagsize, int * const scattered_entries)
    {
	if (bagsize < requiredsize)
	    return;

	assert(blockDim.x * gridDim.x >= np);
	assert(blockDim.x >= 26);

	__shared__ int histo[26];
	__shared__ int base[26];

	const int tid = threadIdx.x; 

	if (tid < 26)
	    histo[tid] = 0;

	__syncthreads();

	int offset[26];
	for(int i = 0; i < 26; ++i)
	    offset[i] = -1;

	Particle p;

	const int pid = tid + blockDim.x * blockIdx.x;

	if (pid < np)
	{
	    p = particles[pid];

	    for(int c = 0; c < 3; ++c)
		assert(p.x[c] >= -L / 2 && p.x[c] < L / 2);

	    for(int i = 0; i < 26; ++i)
	    {
		int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };

		bool halo = true;			

		for(int c = 0; c < 3; ++c)
		{
		    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
		    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

		    const float x = p.x[c];

		    halo &= (x >= halo_start && x < halo_end);
		}

		if (halo)
		    offset[i] = atomicAdd(histo + i, 1);
	    }
	}
	__syncthreads();

	if (tid < 26 && histo[tid] > 0)
	    base[tid] = atomicAdd(global_histo + tid, histo[tid]);

	__syncthreads();

	for(int i = 0; i < 26; ++i)
	    if (offset[i] != -1)
	    {
		const int entry = base[i] + offset[i];
		assert(entry >= 0 && entry < global_histo[26]); 

		bag[ entry ] = p; 
		scattered_entries[ entry ] = pid;
	    }
    }

    __global__ void shift_recv_particles(Particle * p, int n, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	for(int c = 0; c < 3; ++c)
	    assert(p[pid].x[c] >= -L / 2 && p[pid].x[c] < L / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	for(int c = 0; c < 3; ++c)
	    p[pid].x[c] += d[c] * L;

#ifndef NDEBUG

	assert(p[pid].x[0] <= -L / 2 || p[pid].x[0] >= L / 2 ||
	       p[pid].x[1] <= -L / 2 || p[pid].x[1] >= L / 2 || 
	       p[pid].x[2] <= -L / 2 || p[pid].x[2] >= L / 2);

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L - L/2, -L/2 - 1);
	    const float halo_end = min(d[c] * L + L/2, L/2 + 1);
	    const float eps = 1e-5;
	    assert(p[pid].x[c] >= halo_start - eps && p[pid].x[c] <= halo_end + eps);
	}

#endif
    }

    __global__ void check_send_particles(Particle * p, int n, int L, int code)
    {
	assert(blockDim.x * gridDim.x >= n);

	const int pid = threadIdx.x + blockDim.x * blockIdx.x;

	if (pid >= n)
	    return;

	assert(p[pid].x[0] >= -L / 2 || p[pid].x[0] < L / 2 ||
	       p[pid].x[1] >= -L / 2 || p[pid].x[1] < L / 2 || 
	       p[pid].x[2] >= -L / 2 || p[pid].x[2] < L / 2);

	const int d[3] = { (code + 2) % 3 - 1, (code / 3 + 2) % 3 - 1, (code / 9 + 2) % 3 - 1 };

	for(int c = 0; c < 3; ++c)
	{
	    const float halo_start = max(d[c] * L - L/2 - 1, -L/2);
	    const float halo_end = min(d[c] * L + L/2 + 1, L/2);

	    if (!(p[pid].x[c] >= halo_start && p[pid].x[c] < halo_end))
	    {
		printf("oooops particle %d: %e %e %e component %d not within %f , %f\n", pid, p[pid].x[0], p[pid].x[1], p[pid].x[2],
		       c, halo_start, halo_end);
		
	    }
	    const float eps = 1e-5;
	    assert(p[pid].x[c] >= halo_start - eps && p[pid].x[c] < halo_end + eps);
	}
    }
}

void HaloExchanger::pack_and_post(const Particle * const p, const int n, const int * const cellsstart, const int * const cellscount)
{
    nlocal = n;

    for(int i = 0; i < 26; ++i)
    {
	int d[3] = { (i + 2) % 3 - 1, (i / 3 + 2) % 3 - 1, (i / 9 + 2) % 3 - 1 };
	 
	int halo_start[3], halo_size[3];
	for(int c = 0; c < 3; ++c)
	{
	    halo_start[c] = max(d[c] * L - L/2 - 1, -L/2);
	    halo_size[c] = min(d[c] * L + L/2 + 1, L/2) - halo_start[c];
	}

	const int nentries = sendcellstarts[i].size;
	assert(nentries == tmpcount[i].size);
	assert(nentries == tmpstart[i].size);
	    
	//printf("code %d: %d %d %d, nentries %d halo size %d %d %d\n", i, d[0], d[1], d[2], nentries, halo_size[0], halo_size[1], halo_size[2]);
	assert(nentries == 1 + halo_size[0] * halo_size[1] * halo_size[2]);
	
	PackingHalo::count<<< (nentries + 127) / 128, 128 >>>(cellsstart, cellscount, 
							      make_int3(halo_start[0] + L/2 , halo_start[1] + L/2, halo_start[2] + L/2),
							      make_int3(halo_size[0], halo_size[1], halo_size[2]), L,
							      tmpstart[i].data, tmpcount[i].data);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaStreamSynchronize(0));

	/*{
	    thrust::host_vector<int> asd(thrust::device_ptr<int>(tmpcount[i].data),
					 thrust::device_ptr<int>(tmpcount[i].data + nentries));

	    printf("ASD:\n");
	    for(int e = 0; e < tmpcount[i].size; ++e)
		printf("%d ", (int)asd[e]);
	    printf("\n");


	    }*/
	
	thrust::exclusive_scan(thrust::device_ptr<int>(tmpcount[i].data),
			       thrust::device_ptr<int>(tmpcount[i].data + nentries),
			       thrust::device_ptr<int>(sendcellstarts[i].data));
	
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaStreamSynchronize(0));
    stage2:
	
	PackingHalo::fill<<<nentries, 32>>>(p, n, tmpstart[i].data, tmpcount[i].data, L, sendcellstarts[i].data,
					    sendbufs[i].data, sendbufs[i].capacity, scattered_entries[i].data, required_send_bag_size + i, i);

   	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaStreamSynchronize(0));

	const int nrequired = required_send_bag_size_host[i];
/*	{
	    thrust::host_vector<int> asd(thrust::device_ptr<int>(sendcellstarts[i].data),
					 thrust::device_ptr<int>(sendcellstarts[i].data + nentries));
	    // printf("nrequired: %d asd %d\n", nrequired, (int)asd[nentries - 1]);

	    assert(  nrequired   ==   (int)asd[nentries - 1] );
	    }*/

	const int asd = sendbufs[i].size;
	const int asd2 = nrequired;
	const bool fail = sendbufs[i].capacity < nrequired;
	
	if (fail)
	{
	    printf("------------------- rank %d - code %d : oooops %d %d now: %d\n", myrank, i, asd, asd2, sendbufs[i].size);
	    sendbufs[i].resize(nrequired);
	    scattered_entries[i].resize(nrequired);
	   
	    goto stage2;
	}
	else
	{
	   sendbufs[i].size = nrequired;
	   scattered_entries[i].size = nrequired; 
	}

	//send_counts[i] = nrequired;

#ifndef NDEBUG
	
    	const int nd = sendbufs[i].size;

	//assert(nd == send_counts[i]);
	
	if (nd > 0)
	    PackingHalo::check_send_particles<<<(nd + 127)/ 128, 128>>>(sendbufs[i].data, nd, L, i);
#endif

		CUDA_CHECK(cudaPeekAtLastError());
		CUDA_CHECK(cudaStreamSynchronize(0));
    }

    //retrieve recv_counts
    {
	MPI_Request sendcountreq[26];

	for(int i = 0; i < 26; ++i)
	    MPI_CHECK( MPI_Isend(&sendbufs[i].size, 1, MPI_INTEGER, dstranks[i], tagbase_dpd_remote_interactions + i+ 150, cartcomm, sendcountreq + i) );
	
	MPI_Status status;

	int sum = 0;
	for(int i = 0; i < 26; ++i)
	{
	int count;
	    MPI_CHECK( MPI_Recv(&count, 1, MPI_INTEGER, dstranks[i],  recv_tags[i] + 150, cartcomm, &status) );

	    recvbufs[i].resize(count);

	    sum += count;
	}

	MPI_Status culo[26];
	MPI_CHECK( MPI_Waitall(26, sendcountreq, culo) );
    }

    nrecvreq = 0;

    for(int i = 0; i < 26; ++i)
    {
	const int count = recvbufs[i].size;

	if (count == 0)
	    continue;

	MPI_CHECK( MPI_Irecv(recvbufs[i].data, count * 6, /*Particle::datatype()*/ MPI_FLOAT, dstranks[i], 
			     recv_tags[i], cartcomm, recvreq + nrecvreq) );	

	++nrecvreq;
    }

    nsendreq = 0;

    for(int i = 0; i < 26; ++i)
    {
	const int count = sendbufs[i].size;

	if (count == 0) 
	    continue;

	MPI_CHECK( MPI_Isend(sendbufs[i].data, count * 6, /*Particle::datatype()*/ MPI_FLOAT, dstranks[i], 
			     tagbase_dpd_remote_interactions + i, cartcomm, sendreq + nsendreq) );

	++nsendreq;
    }
}

void HaloExchanger::wait_for_messages()
{
    {
	MPI_Status statuses[26];

	MPI_CHECK( MPI_Waitall(nrecvreq, recvreq, statuses) );    
	MPI_CHECK( MPI_Waitall(nsendreq, sendreq, statuses) ); 
    }

    /*for(int i = 0; i < 26; ++i)
    {
	const int count = recv_offsets[i + 1] - recv_offsets[i];

	CUDA_CHECK(cudaMemcpy(recv_bag.data + recv_offsets[i], recvbufs[i].data,  
			      count * sizeof(Particle), cudaMemcpyDeviceToDevice));
    }*/

    for(int i = 0; i < 26; ++i)
    {
	const int ns = recvbufs[i].size;

	if (ns > 0)
	    PackingHalo::shift_recv_particles<<<(ns + 127) / 128, 128>>>(recvbufs[i].data, ns, L, i);	
    }
}

int HaloExchanger::nof_sent_particles()
{
    int s = 0;
    for(int i = 0; i < 26; ++i)
	s += sendbufs[i].size;

    return s;
}

SimpleDeviceBuffer<Particle> HaloExchanger::exchange(const Particle * const plocal, int nlocal,
	    const int * const cellsstart, const int * const cellscount)
{
    pack_and_post(plocal, nlocal, cellsstart, cellscount);
    wait_for_messages();

    int s = 0;
    for(int i = 0; i < 26; ++i)
	s += recvbufs[i].size;
    
    SimpleDeviceBuffer<Particle> retval(s);

    s = 0;
    for(int i = 0; i < 26; ++i)
    {
	CUDA_CHECK(cudaMemcpy(retval.data + s, recvbufs[i].data, recvbufs[i].size * sizeof(Particle), cudaMemcpyDeviceToDevice));
	s += recvbufs[i].size;
    }

    return retval;
}

HaloExchanger::~HaloExchanger()
{    
    CUDA_CHECK(cudaFreeHost(required_send_bag_size));

    MPI_CHECK(MPI_Comm_free(&cartcomm));
}
