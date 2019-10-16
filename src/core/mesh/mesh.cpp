#include "mesh.h"

#include <core/utils/cuda_common.h>

#include <fstream>
#include <vector>

Mesh::Mesh()
{}

Mesh::Mesh(const std::string& fname)
{
    _readOff(fname);
    _check();

    vertexCoordinates.uploadToDevice(defaultStream);
    triangles.uploadToDevice(defaultStream);

    _computeMaxDegree();
}

Mesh::Mesh(const std::vector<float3>& vertices, const std::vector<int3>& faces)
{
    nvertices  = vertices.size();
    ntriangles = faces.size();

    vertexCoordinates.resize_anew(nvertices);
    triangles.resize_anew(ntriangles);

    for (int i = 0; i < ntriangles; ++i)
        triangles[i] = faces[i];

    for (int i = 0; i < nvertices; ++i)
    {
        const float3 v = vertices[i];
        vertexCoordinates[i] = make_float4(v.x, v.y, v.z, 0.f);
    }

    _check();
    
    vertexCoordinates.uploadToDevice(defaultStream);
    triangles.uploadToDevice(defaultStream);

    _computeMaxDegree();
}

Mesh::Mesh(Mesh&&) = default;

Mesh& Mesh::operator=(Mesh&&) = default;

Mesh::~Mesh() = default;

const int& Mesh::getNtriangles() const {return ntriangles;}
const int& Mesh::getNvertices()  const {return nvertices;}

const int& Mesh::getMaxDegree() const {
    if (maxDegree < 0) die("maxDegree was not computed");
    return maxDegree;
}

PyTypes::VectorOfFloat3 Mesh::getVertices()
{
    vertexCoordinates.downloadFromDevice(defaultStream, ContainersSynch::Synch);
    PyTypes::VectorOfFloat3 ret(getNvertices());

    for (int i = 0; i < getNvertices(); ++i) {
        auto r = vertexCoordinates[i];
        ret[i][0] = r.x;
        ret[i][1] = r.y;
        ret[i][2] = r.z;
    }
    return ret;
}

PyTypes::VectorOfInt3 Mesh::getTriangles()
{
    triangles.downloadFromDevice(defaultStream, ContainersSynch::Synch);
    PyTypes::VectorOfInt3 ret(getNtriangles());

    for (int i = 0; i < getNtriangles(); ++i) {
        auto t = triangles[i];
        ret[i][0] = t.x;
        ret[i][1] = t.y;
        ret[i][2] = t.z;
    }
    return ret;
}


void Mesh::_computeMaxDegree()
{
    std::vector<int> degrees(nvertices);

    for (auto t : triangles) {
        degrees[t.x] ++;
        degrees[t.y] ++;
        degrees[t.z] ++;
    }

    maxDegree = *std::max_element(degrees.begin(), degrees.end());
    debug("max degree is %d", maxDegree);
}

void Mesh::_check() const
{
    auto check = [this] (int tr) {
        if (tr < 0 || tr >= nvertices)
            die("Bad triangle indices");
    };

    for (int i = 0; i < getNtriangles(); ++i) {
        check(triangles[i].x);
        check(triangles[i].y);
        check(triangles[i].z);
    }
}

void Mesh::_readOff(const std::string& fname)
{
   std::ifstream fin(fname);
    if (!fin.good())
        die("Mesh file '%s' not found", fname.c_str());

    debug("Reading mesh from file '%s'", fname.c_str());

    std::string line;
    std::getline(fin, line); // OFF header

    int nedges;
    fin >> nvertices >> ntriangles >> nedges;
    std::getline(fin, line); // Finish with this line

    // Read the vertex coordinates
    vertexCoordinates.resize_anew(nvertices);
    for (int i = 0; i < nvertices; i++)
        fin >> vertexCoordinates[i].x >> vertexCoordinates[i].y >> vertexCoordinates[i].z;

    // Read the connectivity data
    triangles.resize_anew(ntriangles);
    for (int i = 0; i < ntriangles; i++)
    {
        int number;
        fin >> number;
        if (number != 3)
            die("Bad mesh file '%s' on line %d, number of face vertices is %d instead of 3",
                    fname.c_str(), 3 /* header */ + nvertices + i, number);

        fin >> triangles[i].x >> triangles[i].y >> triangles[i].z;
    }
}


MeshView::MeshView(const Mesh *m) :
    nvertices  (m->getNvertices()),
    ntriangles (m->getNtriangles()),
    triangles  (m->triangles.devPtr())   
{}
