#include <core/celllist.h>
#include <core/particle_vector.h>
#include <core/helper_math.h>

#include <core/mpi/particle_redistributor.h>
#include <core/mpi/valid_cell.h>

#include <vector>
#include <thread>
#include <algorithm>

__global__ void getExitingParticles(float4* xyzouvwo,
		CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		float4* __restrict__ dests[27], int counts[27])
{
	const int gid = blockIdx.x*blockDim.x + threadIdx.x;
	const int tid = threadIdx.x;
	int cid;
	int cx, cy, cz;

	bool valid = isValidCell(cid, cx, cy, cz, gid, blockIdx.y, cinfo);

	if (!valid) return;

	// The following is called for every outer cell and exactly once for each
	//
	// Now for each cell we check its every particle if it needs to move

	int2 start_size = valid ? cinfo.decodeStartSize(cellsStartSize[cid]) : make_int2(0, 0);

#pragma unroll 2
	for (int i = 0; i < start_size.y; i++)
	{
		const int srcId = start_size.x + i;
		const float4 coo = xyzouvwo[2*srcId];
		const float4 vel = xyzouvwo[2*srcId+1];

		int3 code = cinfo.getCellIdAlongAxis<false>(make_float3(coo));

		if (code.x < 0) code.x = 0;
		else if (code.x >= ncells.x) code.x = 2;
		else code.x = 1;

		if (code.y < 0) code.y = 0;
		else if (code.y >= ncells.y) code.y = 2;
		else code.y = 1;

		if (code.z < 0) code.z = 0;
		else if (code.z >= ncells.z) code.z = 2;
		else code.z = 1;

		if (code.x*code.y*code.z != 1) // this means that the particle has to leave
		{
			const int bufId = (code.z*3 + code.y)*3 + code.x;
			const float4 shift{ cinfo.domainSize.x*(code.x-1),
								cinfo.domainSize.y*(code.y-1),
								cinfo.domainSize.z*(code.z-1), 0 };

			int myid = atomicAdd(counts + bufId, 1);

			const int dstInd = 2*myid;

			float4* addr = dests[bufId];
			float4 newcoo = coo - shift;
			newcoo.w = coo.w;
			addr[dstInd + 0] = newcoo;
			addr[dstInd + 1] = vel;

			// mark the particle as exited to assist cell-list building
			xyzouvwo[2*srcId] = make_float4(-1000.0f, -1000.0f, -1000.0f, coo.w);
		}
	}
}

void ParticleRedistributor::attach(ParticleVector* pv, CellList* cl)
{
	particles.push_back(pv);
	cellLists.push_back(cl);

	const double ndens = (double)pv->np / (cl->ncells.x * cl->ncells.y * cl->ncells.z * cl->rc*cl->rc*cl->rc);
	const int maxdim = std::max({cl->domainSize.x, cl->domainSize.y, cl->domainSize.z});

	// Sizes of buffers. 0 is side, 1 is edge, 2 is corner
	const int sizes[3] = { (int)(ndens * maxdim*maxdim + 128), (int)(ndens * maxdim + 128), (int)(ndens + 128) };

	auto helper = new ExchangeHelper(pv->name, sizes, &pv->halo);
	helpers.push_back(helper);
}

void ParticleRedistributor::redistribute()
{
	init();
	finalize();
}

void ParticleRedistributor::prepareData(int id)
{
	auto pv = particles[id];
	auto cl = cellLists[id];
	auto helper = helpers[id];

	debug2("Preparing %s leaving particles on the device", pv->name.c_str());

	helper->counts.pushStream(defStream);
	helper->counts.clearDevice();
	helper->counts.popStream();

	const int maxdim = std::max({cl->ncells.x, cl->ncells.y, cl->ncells.z});
	const int nthreads = 32;
	if (pv->np > 0)
		getExitingParticles<<< dim3((maxdim*maxdim + nthreads - 1) / nthreads, 6, 1),  dim3(nthreads, 1, 1), 0, helper->stream >>>
					( (float4*)pv->coosvels.devPtr(), cl->cellInfo(), cl->cellsStartSize.devPtr(), helper->sendAddrs.devPtr(), helper->counts.devPtr() );
}

void ParticleHaloExchanger::prepareUploadTarget(int id)
{
	auto pv = particles[id];
	auto helper = helpers[id];

	int oldsize = pv->np;
	pv->resize(oldsize + totalRecvd, resizePreserve);
	helper->target = pv->coosvels.devPtr() + oldsize;
}

void Redistributor::receive(int n)
{
	auto pv = particlesAndCells[n].first;
	auto helper = helpers[n];

	// Wait until messages are arrived
	const int nMessages = helper->requests.size();
	std::vector<MPI_Status> statuses(nMessages);
	MPI_Check( MPI_Waitall(nMessages, &helper->requests[0], &statuses[0]) );

	// Excl scan of message sizes to know where we will upload them
	std::vector<int> offsets(nMessages+1);
	int totalRecvd = 0;
	for (int i=0; i<nMessages; i++)
	{
		offsets[i] = totalRecvd;

		int msize;
		MPI_Check( MPI_Get_count(&statuses[i], mpiPartType, &msize) );
		totalRecvd += msize;
	}
	offsets[nMessages] = totalRecvd;

	int oldsize = pv->np;
	pv->resize(oldsize + totalRecvd, resizePreserve);
	debug2("Receiving %d total %s particles", totalRecvd, pv->name.c_str());

	// Load onto the device
	for (int i=0; i<nMessages; i++)
	{
		debug3("Receiving %s redistribution from rank %d in dircode %d [%2d %2d %2d], size %d",
				pv->name.c_str(), dir2rank[compactedDirs[i]], compactedDirs[i],
				compactedDirs[i]%3 - 1, (compactedDirs[i]/3)%3 - 1, compactedDirs[i]/9 - 1, offsets[i+1] - offsets[i]);

		if (offsets[i+1] - offsets[i] > 0)
			CUDA_Check( cudaMemcpyAsync(pv->coosvels.devPtr() + oldsize + offsets[i], helper->recvBufs[compactedDirs[i]].hostPtr(),
					(offsets[i+1] - offsets[i])*sizeof(Particle), cudaMemcpyHostToDevice, helper->stream) );
	}

	// Reset the cell-list as we've brought in some new particles
	pv->changedStamp++;
	CUDA_Check( cudaStreamSynchronize(helper->stream) );
}


