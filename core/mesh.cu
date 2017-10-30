#include "mesh.h"

#include <fstream>
#include <map>

//Mesh::Mesh(const Mesh& m)
//{
//	nvertices = m.nvertices;
//	ntriangles = m.ntriangles;
//
//	triangles.resize_anew(ntriangles);
//	memcpy(triangles.hostPtr(), m.triangles.hostPtr(), ntriangles * triangles.datatype_size());
//	CUDA_Check( cudaMemcpy(triangles.devPtr(), m.triangles.devPtr(), ntriangles * triangles.datatype_size(), cudaMemcpyDeviceToDevice) );
//}

// Read off mesh
Mesh::Mesh(std::string fname)
{
	std::ifstream fin(fname);
	if (!fin.good())
		die("Mesh file '%s' not found", fname.c_str());

	std::string line;
	std::getline(fin, line); // OFF header

	int nedges;
	fin >> nvertices >> ntriangles >> nedges;

	for (int i=0; i<nvertices; i++)
		std::getline(fin, line);

	triangles.resize_anew(ntriangles);

	for (int i=0; i<ntriangles; i++)
	{
		int number;
		fin >> number;
		if (number != 3)
			die("Bad mesh file '%s' on line %d", fname.c_str(), 3 /* header */ + nvertices + i);

		fin >> triangles[i].x >> triangles[i].y >> triangles[i].z;

		auto check = [&] (int tr) {
			if (tr < 0 || tr >= nvertices);
				die("Bad triangle indices in mesh '%s' on line %d", fname.c_str(), 3 /* header */ + nvertices + i);
		};

		check(triangles[i].x);
		check(triangles[i].y);
		check(triangles[i].z);
	}

	findAdjacent();
}


void Mesh::findAdjacent()
{
	std::vector< std::map<int, int> > adjacentPairs(nvertices);

	for(int i = 0; i < triangles.size(); ++i)
	{
		const int tri[3] = {triangles[i].x, triangles[i].y, triangles[i].z};

		for(int d = 0; d < 3; ++d)
			adjacentPairs[tri[d]][tri[(d + 1) % 3]] = tri[(d + 2) % 3];
	}

	std::vector<int> degrees;
	for(int i = 0; i < nvertices; ++i)
		degrees.push_back(adjacentPairs[i].size());

	auto it = std::max_element(degrees.begin(), degrees.end());
	const int degree = *it;

	if (degree > maxDegree)
		die("Degree of vertex %d is %d > %d (max degree supported)", (int)(it - degrees.begin()), degree, maxDegree);


	// Find first (nearest) neighbors of each vertex
	adjacent.resize_anew(ntriangles * maxDegree);
	for (int i=0; i<adjacent.size(); i++)
		adjacent[i] = -1;

	for(int v = 0; v < nvertices; ++v)
	{
		auto& l = adjacentPairs[v];

		adjacent[0 + maxDegree * v] = l.begin()->first;
		int last = adjacent[1 + maxDegree * v] = l.begin()->second;

		for(int i = 2; i < l.size(); ++i)
		{
			assert(l.find(last) != l.end());

			int tmp = adjacent[i + maxDegree * v] = l.find(last)->second;
			last = tmp;
		}
	}


	// Find distance 2 neighbors of each vertex
	adjacent_second.resize_anew(ntriangles * maxDegree);
	for (int i=0; i<adjacent_second.size(); i++)
		adjacent_second[i] = -1;

	// Get all the vertex neighbors from already compiled adjacent array
	auto extract_neighbors = [&] (const int v) {

		std::vector<int> myneighbors;
		for(int c = 0; c < maxDegree; ++c)
		{
			const int val = adjacent[c + maxDegree * v];
			if (val == -1)
				break;

			myneighbors.push_back(val);
		}

		return myneighbors;
	};

	for(int v = 0; v < nvertices; ++v)
	{
		auto myneighbors = extract_neighbors(v);

		for(int i = 0; i < myneighbors.size(); ++i)
		{
			auto s1 = extract_neighbors(myneighbors[i]);
			std::sort(s1.begin(), s1.end());

			auto s2 = extract_neighbors(myneighbors[(i + 1) % myneighbors.size()]);
			std::sort(s2.begin(), s2.end());

			std::vector<int> result(s1.size() + s2.size());

			const int nterms = std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(),
					result.begin()) - result.begin();

			assert(nterms == 2);

			const int myguy = result[0] == v;

			adjacent_second[i + maxDegree * v] = result[myguy];
		}
	}

	adjacent.uploadToDevice(0);
	adjacent_second.uploadToDevice(0);
}



