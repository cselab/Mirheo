#include "xmf_helpers.h"
#include "common.h"

#include <core/logger.h>

namespace XDMF
{
    namespace XMF
    {        
        void writeDataSet(pugi::xml_node node, std::string h5filename, const Grid* grid, const Channel& channel)
        {            
            auto attrNode = node.append_child("Attribute");
            attrNode.append_attribute("Name") = channel.name.c_str();
            attrNode.append_attribute("AttributeType") = type_to_string(channel.type).c_str();
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
            dataNode.append_attribute("NumberType") = datatypeToString(channel.datatype).c_str();
            dataNode.append_attribute("Precision") = std::to_string(datatypeToPrecision(channel.datatype)).c_str();
            dataNode.append_attribute("Format") = "HDF";
            dataNode.text() = (h5filename + ":/" + channel.name).c_str();
        }
        
        void writeData(pugi::xml_node node, std::string h5filename, const Grid* grid, const std::vector<Channel>& channels)
        {
            for (auto& channel : channels) 
                writeDataSet(node, h5filename, grid, channel);
        }

        static bool is_master_rank(MPI_Comm comm)
        {
            int rank;
            MPI_Check( MPI_Comm_rank(comm, &rank) );
            return (rank == 0);
        }
        
        void write(std::string filename, std::string h5filename, MPI_Comm comm, const Grid *grid, const std::vector<Channel>& channels, float time)
        {
            if (is_master_rank(comm)) {
                pugi::xml_document doc;
                auto root = doc.append_child("Xdmf");
                root.append_attribute("Version") = "3.0";
                auto domain = root.append_child("Domain");

                auto gridNode = grid->write_to_XMF(domain, h5filename);
                
                if (time > -1e-6) gridNode.append_child("Time").append_attribute("Value") = std::to_string(time).c_str();
                writeData(gridNode, h5filename, grid, channels);
                
                doc.save_file(filename.c_str());
            }
            
            MPI_Check( MPI_Barrier(comm) );
        }

        static Channel readDataSet(pugi::xml_node node)
        {
            auto infoNode = node.child("Information");
            auto dataNode = node.child("DataItem");

            std::string name    = node.attribute("Name").value();
            std::string typeStr = infoNode.attribute("Value").value();

            std::string channelType = node.attribute("AttributeType").value();
            auto type = string_to_type (channelType);

            std::string channelDatatype = dataNode.attribute("NumberType").value();
            auto datatype = stringToDatatype(channelDatatype);

            if (type == Channel::Type::Other)
                die("Unrecognised type %s", channelType.c_str());
            
            int entrySize_bytes = get_ncomponents(type) * sizeof(float);

            return Channel(name, nullptr, type, entrySize_bytes, typeStr, datatype);
        }
        
        static void readData(pugi::xml_node node, std::vector<Channel>& channels)
        {
            for (auto attr : node.children("Attribute"))
                if (std::string(attr.name()) == "Attribute")
                    channels.push_back(readDataSet(attr));
        }
        
        void read(std::string filename, MPI_Comm comm, std::string &h5filename, Grid *grid, std::vector<Channel> &channels)
        {
            pugi::xml_document doc;
            auto parseResult = doc.load_file(filename.c_str());

            if (!parseResult)
                die("parsing error while reading '%s'.\n"
                    "\tError description: %s", filename.c_str(), parseResult.description());

            auto gridNode = doc.child("Xdmf").child("Domain").child("Grid");

            grid->read_from_XMF(gridNode, h5filename);

            readData(gridNode, channels);

            MPI_Check( MPI_Barrier(comm) );
        }
    }
}
