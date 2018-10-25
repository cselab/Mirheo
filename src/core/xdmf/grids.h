#pragma once

#include <vector>
#include <string>
#include <memory>
#include <mpi.h>
#include <hdf5.h>

#include <extern/pugixml/src/pugixml.hpp>
#include <cuda_runtime.h>

#include "channel.h"

namespace XDMF
{    
    class Grid
    {
    public:
        virtual std::vector<hsize_t> getLocalSize()                                       const = 0;
        virtual std::vector<hsize_t> getGlobalSize()                                      const = 0;
        virtual std::vector<hsize_t> getOffsets()                                         const = 0;
        virtual bool localEmpty()                                                         const = 0;
        virtual bool globalEmpty()                                                        const = 0;
        virtual int getDims()                                                             const = 0;
                                                                                       
        virtual std::string getCentering()                                                const = 0;
                                                                                       
        virtual void write_to_HDF5(hid_t file_id, MPI_Comm comm)                          const = 0;
        virtual pugi::xml_node write_to_XMF(pugi::xml_node node, std::string h5filename)  const = 0;

        virtual void read_from_XMF(const pugi::xml_node &node, std::string &h5filename)         = 0;
        virtual void split_read_access(MPI_Comm comm, int chunk_size=1)                         = 0;
        virtual void read_from_HDF5(hid_t file_id, MPI_Comm comm)                               = 0;        
        
        virtual ~Grid() = default;
    };
    
    class UniformGrid : public Grid
    {
    public:
        std::vector<hsize_t> getLocalSize()                                       const override;
        std::vector<hsize_t> getGlobalSize()                                      const override;
        std::vector<hsize_t> getOffsets()                                         const override;
        bool localEmpty()                                                         const override;
        bool globalEmpty()                                                        const override;
        int getDims()                                                             const override;
                                                                               
        std::string getCentering()                                                const override;
                                                                               
        void write_to_HDF5(hid_t file_id, MPI_Comm comm)                          const override;
        pugi::xml_node write_to_XMF(pugi::xml_node node, std::string h5filename)  const override;
        
        void read_from_XMF(const pugi::xml_node &node, std::string &h5filename)         override;
        void split_read_access(MPI_Comm comm, int chunk_size = 1)                       override;        
        void read_from_HDF5(hid_t file_id, MPI_Comm comm)                               override;
        
        UniformGrid(int3 localSize, float3 h, MPI_Comm cartComm);
        
    protected:
        std::vector<hsize_t> localSize, globalSize, offsets;
        std::vector<float> spacing;
    };
    
        
    class VertexGrid : public Grid
    {
    public:
        std::vector<hsize_t> getLocalSize()                                       const override;
        std::vector<hsize_t> getGlobalSize()                                      const override;
        std::vector<hsize_t> getOffsets()                                         const override;
        bool localEmpty()                                                         const override;
        bool globalEmpty()                                                        const override;
        int getDims()                                                             const override;
                                                                               
        std::string getCentering()                                                const override;
        std::shared_ptr<std::vector<float>> getPositions()                        const;
                                                                               
        void write_to_HDF5(hid_t file_id, MPI_Comm comm)                          const override;
        pugi::xml_node write_to_XMF(pugi::xml_node node, std::string h5filename)  const override;   
        
        void read_from_XMF(const pugi::xml_node &node, std::string &h5filename)         override;
        void split_read_access(MPI_Comm comm, int chunk_size = 1)                       override;
        void read_from_HDF5(hid_t file_id, MPI_Comm comm)                               override;
        
        VertexGrid(std::shared_ptr<std::vector<float>> positions, MPI_Comm comm);
        
    protected:
        const std::string positionChannelName = "position";
        hsize_t nlocal, nglobal, offset;

        std::shared_ptr<std::vector<float>> positions;

        virtual void _writeTopology(pugi::xml_node& topoNode, std::string h5filename) const;
    };

    class TriangleMeshGrid : public VertexGrid
    {
    public:
        std::shared_ptr<std::vector<int>> getTriangles() const;
        
        void write_to_HDF5(hid_t file_id, MPI_Comm comm)                          const override;
                
        TriangleMeshGrid(std::shared_ptr<std::vector<float>> positions, std::shared_ptr<std::vector<int>> triangles, MPI_Comm comm);
        
    protected:
        const std::string triangleChannelName = "triangle";
        std::shared_ptr<std::vector<int>> triangles;

        void _writeTopology(pugi::xml_node& topoNode, std::string h5filename) const override;
    };
}
