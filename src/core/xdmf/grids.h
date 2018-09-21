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
        virtual std::vector<hsize_t> getLocalSize()                                    const = 0;
        virtual std::vector<hsize_t> getGlobalSize()                                   const = 0;
        virtual std::vector<hsize_t> getOffsets()                                      const = 0;
        virtual bool localEmpty()                                                      const = 0;
        virtual bool globalEmpty()                                                     const = 0;
        virtual int getDims()                                                          const = 0;
                                                                                       
        virtual std::string getCentering()                                             const = 0;
                                                                                       
        virtual void write2HDF5(hid_t file_id, MPI_Comm comm)                          const = 0;
        virtual pugi::xml_node write2XMF(pugi::xml_node node, std::string h5filename)  const = 0;
        
        virtual ~Grid() = default;
    };
    
    class UniformGrid : public Grid
    {
    public:
        std::vector<hsize_t> getLocalSize()                                    const override;
        std::vector<hsize_t> getGlobalSize()                                   const override;
        std::vector<hsize_t> getOffsets()                                      const override;
        bool localEmpty()                                                      const override;
        bool globalEmpty()                                                     const override;
        int getDims()                                                          const override;
                                                                               
        std::string getCentering()                                             const override;
                                                                               
        void write2HDF5(hid_t file_id, MPI_Comm comm)                          const override;
        pugi::xml_node write2XMF(pugi::xml_node node, std::string h5filename)  const override;
        
        UniformGrid(int3 localSize, float3 h, MPI_Comm cartComm);
        
    private:
        std::vector<hsize_t> localSize, globalSize, offsets;
        std::vector<float> spacing;
    };
    
        
    class VertexGrid : public Grid
    {
    public:
        std::vector<hsize_t> getLocalSize()                                    const override;
        std::vector<hsize_t> getGlobalSize()                                   const override;
        std::vector<hsize_t> getOffsets()                                      const override;
        bool localEmpty()                                                      const override;
        bool globalEmpty()                                                     const override;
        int getDims()                                                          const override;
                                                                               
        std::string getCentering()                                             const override;
                                                                               
        void write2HDF5(hid_t file_id, MPI_Comm comm)                          const override;
        pugi::xml_node write2XMF(pugi::xml_node node, std::string h5filename)  const override;   
        
        VertexGrid(int nvertices, const float *positions, MPI_Comm comm);
        
    private:
        const std::string positionChannelName = "position";
        hsize_t nlocal, nglobal, offset;
        const float *positions;
    };
}
