#include "grids.h"
#include "common.h"
#include "hdf5_helpers.h"

#include <core/logger.h>

namespace XDMF
{
    //
    // Uniform Grid
    //
    std::vector<hsize_t> UniformGrid::getLocalSize() const
    {
        return localSize;
    }
    
    std::vector<hsize_t> UniformGrid::getGlobalSize() const
    {
        return globalSize;
    }
    
    std::vector<hsize_t> UniformGrid::getOffsets() const
    {
        return offsets;
    }
    
    bool UniformGrid::localEmpty() const
    {
        int prod = 1;
        for (auto d : localSize) prod *= d;
        
        return (prod == 0);
    }
        
    bool UniformGrid::globalEmpty() const
    {
        int prod = 1;
        for (auto d : globalSize) prod *= d;
        
        return (prod == 0);
    }
    
    int UniformGrid::getDims() const
    {
        return 3;
    }
    
    std::string UniformGrid::getCentering() const
    {
        return "Cell";
    }

    void UniformGrid::write_to_HDF5(hid_t file_id, MPI_Comm comm) const
    {   }
    
    pugi::xml_node UniformGrid::write_to_XMF(pugi::xml_node node, std::string h5filename) const
    {
        auto gridNode = node.append_child("Grid");
        gridNode.append_attribute("Name") = "mesh";
        gridNode.append_attribute("GridType") = "Uniform";
        
        // Topology size is in vertices, so it's +1 wrt to the number of cells
        auto nodeResolution = globalSize;
        for (auto& r : nodeResolution) r += 1;
        
        // One more What. The. F
        std::reverse(nodeResolution.begin(), nodeResolution.end());

        auto topoNode = gridNode.append_child("Topology");
        topoNode.append_attribute("TopologyType") = "3DCORECTMesh";
        topoNode.append_attribute("Dimensions") = to_string(nodeResolution).c_str();
        
        auto geomNode = gridNode.append_child("Geometry");
        geomNode.append_attribute("GeometryType") = "ORIGIN_DXDYDZ";
        
        
        auto setup3Fnode = [] (pugi::xml_node& n, std::string nodeName)
        {
            const std::array< std::pair<std::string, std::string>, 5 > name_vals = {{
                { "Name", nodeName },
                { "Dimensions", "3" },
                { "NumberType", datatypeToString(Channel::Datatype::Float) },
                { "Precision", std::to_string(datatypeToPrecision(Channel::Datatype::Float)) },
                { "Format", "XML" }
            }};
           
            for (auto& name_val : name_vals)
                n.append_attribute(name_val.first.c_str()) = name_val.second.c_str();
        };
        
        auto origNode = geomNode.append_child("DataItem");
        setup3Fnode(origNode, "Origin");
        origNode.text() = "0.0 0.0 0.0";
        
        auto spaceNode = geomNode.append_child("DataItem");
        setup3Fnode(spaceNode, "Spacing");
        spaceNode.text() = to_string(spacing).c_str();
        
        return gridNode;
    }

    void UniformGrid::read_from_XMF(const pugi::xml_node &node, std::string &h5filename)
    {
        // TODO
        die("not implemented");
    }

    void UniformGrid::split_read_access(MPI_Comm comm, int chunk_size)
    {
        // TODO
        die("not implemented");
    }

    void UniformGrid::read_from_HDF5(hid_t file_id, MPI_Comm comm)
    {
        // TODO
        die("not implemented");
    }

    UniformGrid::UniformGrid(int3 localSize, float3 h, MPI_Comm cartComm)
    {
        int nranks[3], periods[3], my3Drank[3];
        MPI_Check( MPI_Cart_get(cartComm, 3, nranks, periods, my3Drank) );
        
        this->spacing    = std::vector<float>{h.x, h.y, h.z};
        this->localSize  = std::vector<hsize_t>{ (hsize_t)localSize.x,  (hsize_t)localSize.y,  (hsize_t)localSize.z};
        
        globalSize = std::vector<hsize_t>{ (hsize_t) nranks[0] * localSize.x,
                                           (hsize_t) nranks[1] * localSize.y,
                                           (hsize_t) nranks[2] * localSize.z};

        offsets = std::vector<hsize_t>{ (hsize_t) my3Drank[2] * localSize.z,
                                        (hsize_t) my3Drank[1] * localSize.y,
                                        (hsize_t) my3Drank[0] * localSize.x,
                                        (hsize_t) 0 };
    }
    
    //
    // Vertex Grid
    //
    
    std::vector<hsize_t> VertexGrid::getLocalSize() const
    {
        return { nlocal };
    }
    
    std::vector<hsize_t> VertexGrid::getGlobalSize() const
    {
        return { nglobal };
    }
    
    std::vector<hsize_t> VertexGrid::getOffsets() const
    {
        return { offset, 0 };
    }
    
    bool VertexGrid::localEmpty() const
    {
        return (nlocal == 0);
    }
        
    bool VertexGrid::globalEmpty() const
    {
        return (nglobal == 0);
    }
    
    int VertexGrid::getDims() const
    {
        return 1;
    }
    
    std::string VertexGrid::getCentering() const
    {
        return "Node";
    }
    
    std::shared_ptr<std::vector<float>> VertexGrid::getPositions() const
    {
        return positions;
    }

    void VertexGrid::write_to_HDF5(hid_t file_id, MPI_Comm comm) const
    {
        Channel posCh(positionChannelName, (void*) positions->data(), Channel::Type::Vector);
        
        HDF5::writeDataSet(file_id, this, posCh);
    }
    
    pugi::xml_node VertexGrid::write_to_XMF(pugi::xml_node node, std::string h5filename) const
    {
        auto gridNode = node.append_child("Grid");
        gridNode.append_attribute("Name") = "mesh";
        gridNode.append_attribute("GridType") = "Uniform";
        
        auto topoNode = gridNode.append_child("Topology");
        topoNode.append_attribute("TopologyType") = "Polyvertex";
        topoNode.append_attribute("NumberOfElements") = std::to_string(nglobal).c_str();
        
        auto geomNode = gridNode.append_child("Geometry");
        geomNode.append_attribute("GeometryType") = "XYZ";
        
        auto partNode = geomNode.append_child("DataItem");
        partNode.append_attribute("Dimensions") = (std::to_string(nglobal) + " 3").c_str();
        partNode.append_attribute("NumberType") = datatypeToString(Channel::Datatype::Float).c_str();
        partNode.append_attribute("Precision") = std::to_string(datatypeToPrecision(Channel::Datatype::Float)).c_str();
        partNode.append_attribute("Format") = "HDF";
        partNode.text() = (h5filename + ":/" + positionChannelName).c_str();
        
        return gridNode;
    }

    void VertexGrid::read_from_XMF(const pugi::xml_node &node, std::string &h5filename)
    {
        int d = 0;

        auto partNode = node.child("Geometry").child("DataItem");

        if (!partNode)
            die("Wrong format");
        
        std::istringstream dimensions( partNode.attribute("Dimensions").value() );

        dimensions >> nglobal;
        dimensions >> d;

        if (d != 3)
            die("expected 3 dimesnional positions, got %d.", d);
        
        std::string positionDataSet(partNode.text().as_string());
        auto endH5 = positionDataSet.find(":");

        if (endH5 == std::string::npos)
            die("expected dataset name from h5 file: got %s", positionDataSet.c_str());

        h5filename = positionDataSet.substr(0, endH5);
    }

    void VertexGrid::split_read_access(MPI_Comm comm, int chunk_size)
    {
        int size, rank;
        int chunk_global, chunk_local, chunk_offset;
        MPI_Check( MPI_Comm_rank(comm, &rank) );
        MPI_Check( MPI_Comm_size(comm, &size) );

        chunk_global = nglobal / chunk_size;
        if (chunk_global * chunk_size != nglobal)
            die("incompatible chunk size");

        chunk_local  = (chunk_global + size - 1) / size;
        chunk_offset = chunk_local * rank;

        if (chunk_offset + chunk_local > chunk_global)
            chunk_local = chunk_global - chunk_offset;

        nlocal = chunk_local  * chunk_size;
        offset = chunk_offset * chunk_size;
    }
    
    void VertexGrid::read_from_HDF5(hid_t file_id, MPI_Comm comm)
    {
        positions->resize(nlocal * 3);
        Channel posCh(positionChannelName, (void*) positions->data(), Channel::Type::Vector);
        
        HDF5::readDataSet(file_id, this, posCh);
    }
        
    VertexGrid::VertexGrid(std::shared_ptr<std::vector<float>> positions, MPI_Comm comm) :
        nlocal(positions->size() / 3), positions(positions)
    {
        if (positions->size() != nlocal * 3)
            die("expected size is multiple of 3; given %d\n", positions->size());
        
        offset = 0;
        MPI_Check( MPI_Exscan   (&nlocal, &offset,  1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
        MPI_Check( MPI_Allreduce(&nlocal, &nglobal, 1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
    }

    //
    // Triangle Mesh Grid
    //
    
    std::shared_ptr<std::vector<int>> TriangleMeshGrid::getTriangles() const
    {
        return triangles;
    }

    void TriangleMeshGrid::write_to_HDF5(hid_t file_id, MPI_Comm comm) const
    {
        VertexGrid::write_to_HDF5(file_id, comm);

        // TODO write triangles; need other sizes
        // Channel triCh(triangleChannelName, (void*) triangle->data(), Channel::Type::Trianle, Channel::Datatype::Int);
        
        // HDF5::writeDataSet(file_id, this, triCh);
    }
                
    TriangleMeshGrid::TriangleMeshGrid(std::shared_ptr<std::vector<float>> positions, std::shared_ptr<std::vector<int>> triangles, MPI_Comm comm) :
        VertexGrid(positions, comm), triangles(triangles)
    {
        if (triangles->size() % 3 != 0)
            die("connectivity: expected size is multiple of 3; given %d\n", triangles->size());
    }
}
