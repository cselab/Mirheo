// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include <mirheo/core/containers.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/utils/common.h>

#include <tuple>
#include <vector>
#include <vector_types.h>

namespace mirheo
{

/** \brief A triangle mesh structure.

    The topology is represented by a list of faces (three vertex indices per face).
 */
class Mesh
{
public:
    /// Default constructor. no vertex and faces.
    Mesh();

    /** Construct a \c Mesh from a off file
        \param fileName The name of the file (contains the extension).
     */
    Mesh(const std::string& fileName);
    /// Construct a \c Mesh from a list of vertices and faces.
    Mesh(const std::tuple<std::vector<real3>, std::vector<int3>>& mesh);
    /// Construct a \c Mesh from a list of vertices and faces.
    Mesh(const std::vector<real3>& vertices, const std::vector<int3>& faces);

    Mesh(Mesh&&); ///< move constructor
    Mesh& operator=(Mesh&&); ///< move assignment operator

    virtual ~Mesh();

    int getNtriangles() const; ///< \return the number of faces
    int getNvertices() const;  ///< \return the number of vertices
    int getMaxDegree() const;  ///< \return the maximum valence of all vertices


    const PinnedBuffer<real4>& getVertices() const; ///< \return the list of vertices
    const PinnedBuffer<int3>& getFaces() const;     ///< \return the list of faces

protected:

    /// Update the internal value maxDegree_ from the current topology.
    void _computeMaxDegree();

    /// Check if all faces contain valid indices; dies otherwise
    void _check() const;

protected:
    PinnedBuffer<int3> faces_; ///< The list of faces
    PinnedBuffer<real4> vertices_; ///< coordinates of all vertices (float4 to reduce number of load instructions)

private:
    int nvertices_{0};  ///< number of vertices
    int ntriangles_{0}; ///< number of faces

    /// max degree of all vertices in the mesh
    int maxDegree_ {-1};
};

/// A device-compatible structure that represents a triangle mesh topology (\see \c Mesh)
struct MeshView
{
    int nvertices;   ///< number of vertices
    int ntriangles;  ///< number of faces
    int3 *triangles; ///< list of faces

    /// Construct a MeshView from a \c Mesh
    MeshView(const Mesh *m);
};

} // namespace mirheo
