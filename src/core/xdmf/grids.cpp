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
    
    void UniformGrid::write2HDF5(hid_t file_id, MPI_Comm comm) const
    {   }
    
    void UniformGrid::write2XMF(pugi::xml_node node, std::string h5filename) const
    {
        // Topology size is in vertices, so it's +1 wrt to the number of cells
        auto nodeResolution = globalSize;
        for (auto& r : nodeResolution) r += 1;
        
        auto topoNode = node.append_child("Topology");
        topoNode.append_attribute("TopologyType") = "3DCORECTMesh";
        topoNode.append_attribute("Dimensions") = to_string(nodeResolution).c_str();
        
        auto geomNode = node.append_child("Geometry");
        geomNode.append_attribute("GeometryType") = "ORIGIN_DXDYDZ";
        
        
        auto setup3Fnode = [] (pugi::xml_node& node, const std::string& name)
        {
            const static std::vector< std::pair<std::string, std::string> > name_vals = {
                { "Name", name }, { "Dimensions", "3" }, { "NumberType", "float" }, { "Precision", "4" }, { "Format", "XML" }
            };
           
            for (auto& name_val : name_vals)
                node.append_attribute(name_val.first.c_str()) = name_val.second.c_str();
        };
        
        auto origNode = geomNode.append_child("DataItem");
        setup3Fnode(origNode, "Origin");
        origNode.text() = "0.0 0.0 0.0";
        
        auto spaceNode = geomNode.append_child("DataItem");
        setup3Fnode(spaceNode, "Spacing");
        spaceNode.text() = to_string(spacing).c_str();
    }
    
    UniformGrid::UniformGrid(int3 localSize, int3 globalSize, float3 h, MPI_Comm cartComm)
    {
        this->localSize  = std::vector<hsize_t>{ (hsize_t)localSize.x,  (hsize_t)localSize.y,  (hsize_t)localSize.z};
        this->globalSize = std::vector<hsize_t>{ (hsize_t)globalSize.x, (hsize_t)globalSize.y, (hsize_t)globalSize.z};
        
        this->spacing = std::vector<float>{h.x, h.y, h.z};
        
        int nranks[3], periods[3], my3Drank[3];
        MPI_Check( MPI_Cart_get(cartComm, 3, nranks, periods, my3Drank) );
        
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
    
    void VertexGrid::write2HDF5(hid_t file_id, MPI_Comm comm) const
    {
        Channel posCh(positionChannelName, (void*)positions, Channel::Type::Vector, 3*sizeof(float), "float3");
        
        HDF5::writeDataSet(file_id, this, posCh);
    }
    
    void VertexGrid::write2XMF(pugi::xml_node node, std::string h5filename) const
    {        
        auto topoNode = node.append_child("Topology");
        topoNode.append_attribute("TopologyType") = "Polyvertex";
        topoNode.append_attribute("NumberOfElements") = std::to_string(nglobal).c_str();
        
        auto geomNode = node.append_child("Geometry");
        geomNode.append_attribute("GeometryType") = "XYZ";
        
        auto partNode = geomNode.append_child("DataItem");
        partNode.append_attribute("Dimensions") = (std::to_string(nglobal) + " 3").c_str();
        partNode.append_attribute("NumberType") = "float";
        partNode.append_attribute("Precision") = "4";
        partNode.append_attribute("Format") = "HDF";
        partNode.text() = (h5filename + ":/" + positionChannelName).c_str();
    }
    
    VertexGrid::VertexGrid(int nvertices, const float *positions, MPI_Comm comm) :
        nlocal(nvertices), positions(positions)
    {
        offset = 0;
        MPI_Check( MPI_Exscan   (&nlocal, &offset,  1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
        MPI_Check( MPI_Allreduce(&nlocal, &nglobal, 1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
    }
}
