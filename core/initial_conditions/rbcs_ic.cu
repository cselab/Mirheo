#include "rbcs_ic.h"

#include <random>
#include <fstream>

#include <core/pvs/particle_vector.h>
#include <core/pvs/rbc_vector.h>
#include <core/rigid_kernels/integration.h>

void RBC_IC::readVertices(std::string fname, PinnedBuffer<float4>& positions)
{
	std::ifstream fin(fname);
	if (!fin.good())
		die("Mesh file '%s' not found", fname.c_str());

	std::string line;
	std::getline(fin, line); // OFF header

	int nvertices, ntriangles, nedges;
	fin >> nvertices >> ntriangles >> nedges;

	positions.resize_anew(nvertices);
	for (int i=0; i<nvertices; i++)
		fin >> positions[i].x >> positions[i].y >> positions[i].z;
}

void RBC_IC::exec(const MPI_Comm& comm, ParticleVector* pv, DomainInfo domain, cudaStream_t stream)
{
	auto ov = dynamic_cast<RBCvector*>(pv);
	if (ov == nullptr)
		die("RBCs can only be generated out of rbc object vectors");

	pv->domain = domain;

	PinnedBuffer<float4> vertices;
	readVertices(offfname, vertices);
	if (ov->objSize != vertices.size())
		die("Object size and number of vertices in mesh match for %s", ov->name.c_str());

	std::ifstream fic(icfname);
	int nObjs=0;

	while (true)
	{
		float3 com;
		float4 q;

		fic >> com.x >> com.y >> com.z;
		fic >> q.x >> q.y >> q.z >> q.w;

		if (fic.fail()) break;

		q = normalize(q);

		if (ov->domain.globalStart.x <= com.x && com.x < ov->domain.globalStart.x + ov->domain.localSize.x &&
		    ov->domain.globalStart.y <= com.y && com.y < ov->domain.globalStart.y + ov->domain.localSize.y &&
		    ov->domain.globalStart.z <= com.z && com.z < ov->domain.globalStart.z + ov->domain.localSize.z)
		{
			com = domain.global2local(com);
			int oldSize = ov->local()->size();
			ov->local()->resize(oldSize + vertices.size(), stream);

			for (int i=0; i<vertices.size(); i++)
			{
				float3 r = rotate(f4tof3(vertices[i]), q) + com;
				Particle p;
				p.r = r;
				p.u = make_float3(0);

				ov->local()->coosvels[oldSize + i] = p;
			}

			nObjs++;
		}
	}

	// Set ids
	int totalCount=0; // TODO: int64!
	MPI_Check( MPI_Exscan(&nObjs, &totalCount, 1, MPI_INT, MPI_SUM, comm) );

	auto ids = ov->local()->getDataPerObject<int>("ids");
	for (int i=0; i<nObjs; i++)
		(*ids)[i] = totalCount + i;

	for (int i=0; i < ov->local()->size(); i++)
		ov->local()->coosvels[i].i1 = totalCount*ov->objSize + i;


	ids->uploadToDevice(stream);
	ov->local()->coosvels.uploadToDevice(stream);

	info("Read %d %s rbcs", nObjs, ov->name.c_str());
}

