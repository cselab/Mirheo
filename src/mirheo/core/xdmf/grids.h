// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "channel.h"

#include <extern/pugixml/src/pugixml.hpp>

#include <cuda_runtime.h>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

namespace mirheo
{

namespace XDMF
{

/** \brief Interface to represent the dimensions of the geometry data
 */
class GridDims
{
public:
    virtual ~GridDims() = default;

    virtual std::vector<hsize_t> getLocalSize()  const = 0; ///< number of elements in the current subdomain
    virtual std::vector<hsize_t> getGlobalSize() const = 0; ///< number of elements in the whole domain
    virtual std::vector<hsize_t> getOffsets()    const = 0; ///< start indices in the current subdomain

    bool localEmpty()   const; ///< \return \c true if there is no data in the current subdomain
    bool globalEmpty()  const; ///< \return \c true if there is no data in the whole domain
    int getDims()       const; ///< \return The current dimension of the data (e.g. 3D for uniform grids, 1D for particles)
};

/** \brief Interface to represent The geometry of channels to dump
 */
class Grid
{
public:
    virtual ~Grid() = default;

    virtual const GridDims* getGridDims() const = 0; ///< \return the GridDims that describes the data dimensions
    virtual std::string getCentering() const = 0;    ///< \return A string describing (for XDMF) data location (e.g. "Node" or "Cell")

    /** Dump the geometry description to hdf5 file
        \param file_id The hdf5 file description
        \param comm MPI communicator that was used to open the file
     */
    virtual void writeToHDF5(hid_t file_id, MPI_Comm comm) const = 0;

    /** Dump the geometry description to xdmf file
        \param node The xml node that will store the geometry information
        \param h5filename name of the hdf5 file that will contain the data
     */
    virtual pugi::xml_node writeToXMF(pugi::xml_node node, std::string h5filename) const = 0;

    /** \brief read the geometry info contained in the xdmf file
        \param [in] node The xmf data
        \param [out] h5filename The name of the associated hdf5 file
        \note must be called before splitReadAccess()
     */
    virtual void readFromXMF(const pugi::xml_node &node, std::string &h5filename) = 0;

    /** \brief Set the number of elements to read for the current subdomain
        \param comm Communicator that will be used to read the hdf5 file
        \param chunkSize For particles, this affects the number of particles to keep together
                         on a single rank. Useful for objects.
        \note must be called after readFromXMF()
     */
    virtual void splitReadAccess(MPI_Comm comm, int chunkSize = 1) = 0;

    /** \brief Read the geometry data contained in the hdf5 file
        \param file_id The hdf5 file reference
        \param comm MPI communicator used in the I/O
        \note must be called after splitReadAccess()
     */
    virtual void readFromHDF5(hid_t file_id, MPI_Comm comm) = 0;
};

/** \brief Representation of a uniform grid geometry.
    Each subdomain has the same number of grid points in every direction.
 */
class UniformGrid : public Grid
{
public:
    /** \brief construct a UniformGrid object
        \param localSize The dimensions of the grid per rank
        \param h grid spacing
        \param cartComm The cartesian communicator that will be used for I/O
        \note all these parameters must be the same on every rank
     */
    UniformGrid(int3 localSize, real3 h, MPI_Comm cartComm);

    const GridDims* getGridDims() const override;
    std::string getCentering()    const override;

    void writeToHDF5(hid_t file_id, MPI_Comm comm)                          const override;
    pugi::xml_node writeToXMF(pugi::xml_node node, std::string h5filename)  const override;

    void readFromXMF(const pugi::xml_node &node, std::string &h5filename) override;
    void splitReadAccess(MPI_Comm comm, int chunkSize = 1)                override;
    void readFromHDF5(hid_t file_id, MPI_Comm comm)                       override;

private:
    class UniformGridDims : public GridDims
    {
    public:
        UniformGridDims(int3 localSize, MPI_Comm cartComm);

        std::vector<hsize_t> getLocalSize()  const override;
        std::vector<hsize_t> getGlobalSize() const override;
        std::vector<hsize_t> getOffsets()    const override;

    private:
        std::vector<hsize_t> localSize_;
        std::vector<hsize_t> globalSize_;
        std::vector<hsize_t> offsets_;
    };

private:
    UniformGridDims dims_;
    std::vector<real> spacing_;
};

/** \brief Representation of particles geometry.

    Each rank contains the positions of the particles in GLOBAL coordinates.
 */
class VertexGrid : public Grid
{
public:
    /** \brief Construct a VertexGrid object
        \param positions The positions of the particles in the current subdomain, in global coordinates
        \param comm The communicator that will be used for I/O
        \note The \p positions are passed as a shared pointer so that this class is able to either
        allocate its own memory or can share it with someone else
     */
    VertexGrid(std::shared_ptr<std::vector<real3>> positions, MPI_Comm comm);

    const GridDims* getGridDims() const override;
    std::string getCentering()    const override;

    void writeToHDF5(hid_t file_id, MPI_Comm comm)                          const override;
    pugi::xml_node writeToXMF(pugi::xml_node node, std::string h5filename)  const override;

    void readFromXMF(const pugi::xml_node &node, std::string &h5filename)         override;
    void splitReadAccess(MPI_Comm comm, int chunkSize = 1)                        override;
    void readFromHDF5(hid_t file_id, MPI_Comm comm)                               override;

protected:
    /// dimensions of the vertex geometry representation
    class VertexGridDims : public GridDims
    {
    public:
        /** \param nLocal number of particles on the current rank
            \param comm communicator
         */
        VertexGridDims(long nLocal, MPI_Comm comm);

        std::vector<hsize_t> getLocalSize()  const override;
        std::vector<hsize_t> getGlobalSize() const override;
        std::vector<hsize_t> getOffsets()    const override;

        hsize_t getNLocal()  const; ///< \return the number of vertices on the current rank
        void setNLocal(hsize_t n);  ///< setthe number of vertices on the current rank

        hsize_t getNGlobal() const;  ///< \return the total number of vertices
        void setNGlobal(hsize_t n);  ///< set the total number of vertices

        void setOffset(hsize_t n);  ///< set the number of vertices present on the "previous" ranks

    private:
        hsize_t nLocal_, nGlobal_, offset_;
    };

private:
    static const std::string positionChannelName_;
    VertexGridDims dims_;

    std::shared_ptr<std::vector<real3>> positions_;

    virtual void _writeTopology(pugi::xml_node& topoNode, const std::string& h5filename) const;
};


/** \brief Representation of triangle mesh geometry.

    This is a VertexGrid associated with the additional connectivity (list of triangle faces).
    The vertices are stored in global coordinates and the connectivity also stores indices in global coordinates.
 */
class TriangleMeshGrid : public VertexGrid
{
public:
    /** \brief Construct a TriangleMeshGrid object
        \param positions The positions of the particles in the current subdomain, in global coordinates
        \param triangles The list of faces in the current subdomain (global indices)
        \param comm The communicator that will be used for I/O
     */
    TriangleMeshGrid(std::shared_ptr<std::vector<real3>> positions, std::shared_ptr<std::vector<int3>> triangles, MPI_Comm comm);

    void writeToHDF5(hid_t file_id, MPI_Comm comm) const override;

private:
    static const std::string triangleChannelName_;
    VertexGridDims dimsTriangles_;
    std::shared_ptr<std::vector<int3>> triangles_;

    void _writeTopology(pugi::xml_node& topoNode, const std::string& h5filename) const override;
};

} // namespace XDMF

} // namespace mirheo
