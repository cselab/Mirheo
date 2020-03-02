#pragma once

#include "mesh.h"

#include <mirheo/core/containers.h>

namespace mirheo
{

/** \brief A triangle mesh with face connectivity, adjacent vertices and geometric precomputed values.
    This class was designed to assist MembraneInteraction.

    A stress-free state can be associated to the mesh.
    The precomputed geometric quantities that are stored in the object are computed from the stress free state.

    Additionally to the list of faces (\see Mesh), this class contains a list of 
    adjacent vertices for each vertex.
    The list is stored in a single array, each vertex having a contiguous chunk of length maxDegree.
    See developer docs for more information.
 */
class MembraneMesh : public Mesh
{
public:
    friend class MembraneMeshView;
    
    /// construct an empty mesh
    MembraneMesh();

    /** \brief Construct a MembraneMesh from an off file
        \param initialMesh File (in off format) that contains the mesh information.
        \note The stress free state will be the one given by \p initialMesh
     */
    MembraneMesh(const std::string& initialMesh);

    /** \brief Construct a MembraneMesh from an off file
        \param initialMesh File (in off format) that contains the mesh information.
        \param stressFreeMesh File (in off format) that contains the stress free state of the mesh.
        \note \p initialMesh and \p stressFreeMesh must have the same topology.
    */
    MembraneMesh(const std::string& initialMesh, const std::string& stressFreeMesh);

    /** \brief Construct a MembraneMesh from a list of vertices and faces
        \param vertices The vertex coordinates of the mesh
        \param faces List of faces that contains the vertex indices.
        \note The stress free state is the same as the current mesh.
     */
    MembraneMesh(const std::vector<real3>& vertices,
                 const std::vector<int3>& faces);
    
    /** \brief Construct a MembraneMesh from a list of vertices and faces
        \param vertices The vertex coordinates of the mesh
        \param stressFreeVertices The vertex coordinates that represent the stress free state.
        \param faces List of faces that contains the vertex indices.
     */
    MembraneMesh(const std::vector<real3>& vertices,
                 const std::vector<real3>& stressFreeVertices,
                 const std::vector<int3>& faces);

    /** \brief Construct a membrane mesh from its snapshot. Reads both initial and stress-free mesh.
        \param [in] loader The \c Loader object. Provides load context and unserialization functions.
        \param [in] config The mesh parameters.
     */
    MembraneMesh(Loader& loader, const ConfigObject& config);

    MembraneMesh(MembraneMesh&&); ///< move constructor
    MembraneMesh& operator=(MembraneMesh&&); ///< move assignment operator
    ~MembraneMesh();

    /** \brief Dump the initial and the stress-free mesh in an .off file, create a ConfigObject with the mesh name and register it in the saver.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.

        Checks that the object type is exactly MembraneMesh.
      */
    void saveSnapshotAndRegister(Saver& saver) override;

protected:
    /** \brief Implementation of the snapshot saving. Reusable by potential derived classes.
        \param [in,out] saver The \c Saver object. Provides save context and serialization functions.
        \param [in] typeName The name of the type being saved.
      */
    ConfigObject _saveSnapshot(Saver& saver, const std::string& typeName);

private:
    /// compute the adjacent vertices lists of all vertices
    void _findAdjacent();

    /// compute the stress free information from the given vertices
    void _computeInitialQuantities(const PinnedBuffer<real4>& vertices);
    /// compute the edge lengths of the stress-free state
    void _computeInitialLengths(const PinnedBuffer<real4>& vertices);
    /// compute the areas of the stress-free state
    void _computeInitialAreas(const PinnedBuffer<real4>& vertices);
    /// compute the dot product between adjacent edges from the stress-free state
    /// used in Lim to determine if cos(phi) < 0
    void _computeInitialDotProducts(const PinnedBuffer<real4>& vertices); /// 

private:
    PinnedBuffer<int> adjacent_; ///< list of adjacent vertices for each vertex
    PinnedBuffer<int> degrees_;  ///< degree (or valence) of each vertex
    PinnedBuffer<real> initialLengths_; ///< length of each edge in the stress-free state; data layout is the same as adjacent_
    PinnedBuffer<real> initialAreas_;    ///< length of each triangle in the stress-free state; data layout is the same as adjacent_
    PinnedBuffer<real> initialDotProducts_;  ///< dot product between two consecutive edges in the stress-free state; data layout is the same as adjacent_
};

/// A device-compatible structure that represents a data stored in a MembraneMesh additionally to its topology
struct MembraneMeshView : public MeshView
{
    int maxDegree; ///< maximum degree of all vertices

    int *adjacent; ///< lists of adjacent vertices
    int *degrees;  ///< degree of each vertex

    real *initialLengths;     ///< lengths of edges in the stress-free state
    real *initialAreas;       ///< areas of each face in the stress-free state
    real *initialDotProducts; ///< do products between adjacent edges in the stress-free state

    /// Construct a MembraneMeshView from a MembraneMesh object
    MembraneMeshView(const MembraneMesh *m);
};

} // namespace mirheo
