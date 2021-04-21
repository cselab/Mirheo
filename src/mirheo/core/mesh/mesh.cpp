// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mesh.h"

#include <mirheo/core/mesh/off.h>
#include <mirheo/core/snapshot.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/helper_math.h>
#include <mirheo/core/utils/path.h>

#include <algorithm>
#include <cassert>
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

Mesh::Mesh(Loader& loader, const ConfigObject& config) :
    Mesh(joinPaths(loader.getContext().getPath(), config["name"] + ".off"))
{}

Mesh::Mesh(const std::vector<real3>& vertices, const std::vector<int3>& faces) :
    nvertices_ (static_cast<int>(vertices.size())),
    ntriangles_(static_cast<int>(faces   .size()))
{
    vertices_.resize_anew(nvertices_);
    faces_   .resize_anew(ntriangles_);

    for (int i = 0; i < ntriangles_; ++i)
        faces_[i] = faces[i];

    for (int i = 0; i < nvertices_; ++i)
    {
        const real3 v = vertices[i];
        vertices_[i] = make_real4(v.x, v.y, v.z, 0.0_r);
    }

    _check();

    vertices_.uploadToDevice(defaultStream);
    faces_   .uploadToDevice(defaultStream);

    _computeMaxDegree();
}

Mesh::Mesh(Mesh&&) = default;

Mesh& Mesh::operator=(Mesh&&) = default;

Mesh::~Mesh() = default;

int Mesh::getNtriangles() const
{
    return ntriangles_;
}

int Mesh::getNvertices()  const
{
    return nvertices_;
}

int Mesh::getMaxDegree() const
{
    if (maxDegree_ < 0)
        die("maxDegree was not computed");
    return maxDegree_;
}

const PinnedBuffer<real4>& Mesh::getVertices() const
{
    return vertices_;
}

const PinnedBuffer<int3>& Mesh::getFaces() const
{
    return faces_;
}

void Mesh::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<Mesh>(this, _saveSnapshot(saver, "Mesh"));
}

ConfigObject Mesh::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    // Increment the "mesh" context counter and the old value as an ID.
    int id = saver.getContext().counters["mesh"]++;
    std::string name = "mesh_" + std::to_string(id);

    if (saver.getContext().isGroupMasterTask()) {
        // Dump the mesh to a file.
        std::string fileName = joinPaths(saver.getContext().path, name + ".off");
        std::vector<int3> tmpTriangles(faces_.begin(), faces_.end());
        std::vector<real3> tmpVertices(vertices_.size());
        for (size_t i = 0; i < tmpVertices.size(); ++i) {
            tmpVertices[i].x = vertices_[i].x;
            tmpVertices[i].y = vertices_[i].y;
            tmpVertices[i].z = vertices_[i].z;
        }
        writeOff(tmpVertices, tmpTriangles, fileName);
    }

    // Note: The mesh file name can be constructed from the object name.
    return ConfigObject{
        {"__category", saver("Mesh")},
        {"__type",     saver(typeName)},
        {"name",       saver(name)},
    };
}

void Mesh::_computeMaxDegree()
{
    std::vector<int> degrees(nvertices_);

    for (auto t : faces_)
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
        check(faces_[i].x);
        check(faces_[i].y);
        check(faces_[i].z);
    }
}

MeshView::MeshView(const Mesh *m) :
    nvertices  (m->getNvertices()),
    ntriangles (m->getNtriangles()),
    triangles  (m->getFaces().devPtr())
{}

} // namespace mirheo
