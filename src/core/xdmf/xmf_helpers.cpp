#include "xmf_helpers.h"
#include "common.h"

#include <core/logger.h>

namespace XDMF
{
    namespace XMF
    {
        std::string to_string(Channel::Type type)
        {
            switch (type)
            {
                case Channel::Type::Scalar:  return "Scalar";
                case Channel::Type::Vector:  return "Vector";
                case Channel::Type::Tensor6: return "Tensor6";
                case Channel::Type::Tensor9: return "Tensor9";
                case Channel::Type::Other:   return "Scalar";
            }
        }
        
        void writeDataSet(pugi::xml_node node, std::string h5filename, const Grid* grid, const Channel& channel)
        {            
            auto attrNode = node.append_child("Attribute");
            attrNode.append_attribute("Name") = channel.name.c_str();
            attrNode.append_attribute("AttributeType") = to_string(channel.type).c_str();
            attrNode.append_attribute("Center") = grid->getCentering().c_str();
            
            // Write type information
            auto infoNode = attrNode.append_child("Information");
            infoNode.append_attribute("Name") = "Typeinfo";
            infoNode.append_attribute("Value") = channel.typeStr.c_str();
            
            // Add one more dimension: number of floats per data item
            auto globalSize = grid->getGlobalSize();
            globalSize.push_back(channel.entrySize_floats);
            
            auto dataNode = attrNode.append_child("DataItem");
            dataNode.append_attribute("Dimensions") = ::to_string(globalSize).c_str();
            dataNode.append_attribute("NumberType") = "Float";
            dataNode.append_attribute("Precision") = "4";
            dataNode.append_attribute("Format") = "HDF";
            dataNode.text() = (h5filename + ":/" + channel.name).c_str();
        }
        
        void writeData(pugi::xml_node node, std::string h5filename, const Grid* grid, const std::vector<Channel>& channels)
        {
            for (auto& channel : channels) 
                writeDataSet(node, h5filename, grid, channel);
        }
        
        void write(std::string filename, std::string h5filename, MPI_Comm comm, const Grid* grid, const std::vector<Channel>& channels, float time)
        {
            int rank;
            MPI_Check( MPI_Comm_rank(comm, &rank) );
            if (rank == 0)
            {
                pugi::xml_document doc;
                auto root = doc.append_child("Xdmf");
                root.append_attribute("Version") = "3.0";
                auto domain = root.append_child("Domain");

                auto gridNode = grid->write2XMF(domain, h5filename);
                
                gridNode.append_child("Time").append_attribute("Value") = std::to_string(time).c_str();
                writeData(gridNode, h5filename, grid, channels);
                
                doc.save_file(filename.c_str());
            }
            
            MPI_Check( MPI_Barrier(comm) );
        }

    }
}
