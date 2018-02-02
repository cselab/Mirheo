#pragma once

#include <core/containers.h>
#include <core/datatypes.h>

class Mesh
{
public:
	// max degree of a vertex in mesh
	// only used in RBC forces
	static const int maxDegree = 7;

	int nvertices{0}, ntriangles{0};

	PinnedBuffer<int3> triangles;
	PinnedBuffer<int> adjacentTriangles;
	PinnedBuffer<int> adjacent, adjacent_second, degrees;

	PinnedBuffer<float4> vertexCoordinates;

	Mesh() {};
	Mesh(std::string);

	Mesh(Mesh&&) = default;
	Mesh& operator=(Mesh&&) = default;

	void findAdjacent();
};

struct MeshView
{
	int nvertices, ntriangles, maxDegree;

	int3* triangles;
	int* adjacentTriangles;
	int *adjacent, *adjacent_second, *degrees;


	MeshView(const Mesh& m)
	{
		nvertices = m.nvertices;
		ntriangles = m.ntriangles;
		maxDegree = m.maxDegree;

		adjacentTriangles = m.adjacentTriangles.devPtr();
		triangles = m.triangles.devPtr();
		adjacent = m.adjacent.devPtr();
		adjacent_second = m.adjacent_second.devPtr();
		degrees = m.degrees.devPtr();
	}
};
