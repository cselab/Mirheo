#pragma once

#include <core/containers.h>
#include <core/datatypes.h>

class Mesh
{
public:
	static const int maxDegree = 7;
	int nvertices{0}, ntriangles{0};

	PinnedBuffer<int3> triangles;
	PinnedBuffer<int> adjacent, adjacent_second;

	Mesh() {};
	Mesh(std::string);

	Mesh(Mesh&&) = default;

private:
	void findAdjacent();
};

struct MeshView
{
	int nvertices, ntriangles, maxDegree;

	int3* triangles;
	int *adjacent, *adjacent_second;

	float4* vertices;

	MeshView(const Mesh& m, PinnedBuffer<Particle>* vertexCoosVels)
	{
		nvertices = m.nvertices;
		ntriangles = m.ntriangles;
		maxDegree = m.maxDegree;

		triangles = m.triangles.devPtr();
		adjacent = m.adjacent.devPtr();
		adjacent_second = m.adjacent_second.devPtr();
		vertices = reinterpret_cast<float4*>(vertexCoosVels->devPtr());
	}
};
