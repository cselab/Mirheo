#include "mesh.h"

#include <mirheo/core/mesh/off.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/helper_math.h>

#include <algorithm>
#include <vector>

namespace mirheo
{

Mesh::Mesh()
{}

Mesh::Mesh(const std::string& fileName) :
    Mesh(readOff(fileName))
{}

Mesh::Mesh(const std::tuple<std::vector<real3>, std::vector<int3>>& mesh) :
    Mesh(std::get<0>(mesh), std::get<1>(mesh))
{}

Mesh::Mesh(const std::vector<real3>& vertices, const std::vector<int3>& faces) :
    nvertices_ (static_cast<int>(vertices.size())),
    ntriangles_(static_cast<int>(faces   .size()))
{
    vertexCoordinates.resize_anew(nvertices_);
    triangles.resize_anew(ntriangles_);

    for (int i = 0; i < ntriangles_; ++i)
        triangles[i] = faces[i];

    for (int i = 0; i < nvertices_; ++i)
    {
        const real3 v = vertices[i];
        vertexCoordinates[i] = make_real4(v.x, v.y, v.z, 0.0_r);
    }

    _check();
    
    vertexCoordinates.uploadToDevice(defaultStream);
    triangles.uploadToDevice(defaultStream);

    _computeMaxDegree();
}

Mesh::Mesh(Mesh&&) = default;

Mesh& Mesh::operator=(Mesh&&) = default;

Mesh::~Mesh() = default;

const int& Mesh::getNtriangles() const {return ntriangles_;}
const int& Mesh::getNvertices()  const {return nvertices_;}

const int& Mesh::getMaxDegree() const {
    if (maxDegree_ < 0)
        die("maxDegree was not computed");
    return maxDegree_;
}

PyTypes::VectorOfReal3 Mesh::getVertices()
{
    vertexCoordinates.downloadFromDevice(defaultStream, ContainersSynch::Synch);
    PyTypes::VectorOfReal3 ret(getNvertices());

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
    std::vector<int> degrees(nvertices_);

    for (auto t : triangles)
    {
        degrees[t.x] ++;
        degrees[t.y] ++;
        degrees[t.z] ++;
    }

    maxDegree_ = *std::max_element(degrees.begin(), degrees.end());
    debug("max degree is %d", maxDegree_);
}

void Mesh::_check() const
{
    auto check = [this] (int tr)
    {
        if (tr < 0 || tr >= nvertices_)
            die("Bad triangle indices");
    };

    for (int i = 0; i < getNtriangles(); ++i)
    {
        check(triangles[i].x);
        check(triangles[i].y);
        check(triangles[i].z);
    }
}

MeshView::MeshView(const Mesh *m) :
    nvertices  (m->getNvertices()),
    ntriangles (m->getNtriangles()),
    triangles  (m->triangles.devPtr())   
{}

Config ConfigDumper<Mesh>::dump(Dumper& dumper, const Mesh& mesh)
{
    // Increment the "mesh" context counter and the old value as an ID.
    int id = dumper.getContext().counters["mesh"]++;
    std::string name = "mesh_" + std::to_string(id);

    if (dumper.getContext().isGroupMasterTask()) {
        // Dump the mesh to a file.
        std::string fileName = joinPaths(dumper.getContext().path, name + ".off");
        std::vector<int3> triangles(mesh.triangles.begin(), mesh.triangles.end());
        std::vector<real3> vertices(mesh.vertexCoordinates.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            vertices[i].x = mesh.vertexCoordinates[i].x;
            vertices[i].y = mesh.vertexCoordinates[i].y;
            vertices[i].z = mesh.vertexCoordinates[i].z;
        }
        writeOff(vertices, triangles, fileName);
    }

    // Only return the name, from which the file name can be constructed.
    return dumper.registerObject(&mesh, Config::Dictionary{
        {"__category", dumper("Mesh")},
        {"__type",     dumper("Mesh")},
        {"name",       dumper(name)},
    });
}

} // namespace mirheo
