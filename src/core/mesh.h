#pragma once

#include <core/containers.h>
#include <core/datatypes.h>

class Mesh
{
public:
	int nvertices{0}, ntriangles{0};

	PinnedBuffer<int3> triangles;
	//PinnedBuffer<int> adjacentTriangles;

	PinnedBuffer<float4> vertexCoordinates;

	Mesh() {};
	Mesh(std::string);

	Mesh(Mesh&&) = default;
	Mesh& operator=(Mesh&&) = default;
};

class MembraneMesh : public Mesh
{
public:
	// max degree of a vertex in mesh
	static const int maxDegree = 7;

	PinnedBuffer<int> adjacent, adjacent_second, degrees;
	PinnedBuffer<float> initialLengths;

	MembraneMesh() {};
	MembraneMesh(std::string);

	MembraneMesh(MembraneMesh&&) = default;
	MembraneMesh& operator=(MembraneMesh&&) = default;

	void findAdjacent();
};



struct MeshView
{
	int nvertices, ntriangles;
	int3* triangles;

	MeshView(const Mesh* m)
	{
		nvertices = m->nvertices;
		ntriangles = m->ntriangles;

		triangles = m->triangles.devPtr();
	}
};

struct MembraneMeshView : public MeshView
{
	int maxDegree;

	int *adjacent, *adjacent_second, *degrees;
	float* initialLengths;

	MembraneMeshView(const MembraneMesh* m) : MeshView(m)
	{
		maxDegree = m->maxDegree;

		adjacent = m->adjacent.devPtr();
		adjacent_second = m->adjacent_second.devPtr();
		degrees = m->degrees.devPtr();
		initialLengths = m->initialLengths.devPtr();
	}
};


