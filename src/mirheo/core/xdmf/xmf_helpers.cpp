#include "xmf_helpers.h"
#include "common.h"

#include <mirheo/core/logger.h>

namespace mirheo
{

namespace XDMF
{
namespace XMF
{        
void writeDataSet(pugi::xml_node node, const std::string& h5filename,
                  const Grid *grid, const Channel& channel)
{            
    auto attrNode = node.append_child("Attribute");
    attrNode.append_attribute("Name") = channel.name.c_str();
    attrNode.append_attribute("AttributeType") = dataFormToXDMFAttribute(channel.dataForm).c_str();
    attrNode.append_attribute("Center") = grid->getCentering().c_str();
            
    // Write type information
    auto infoNode = attrNode.append_child("Information");
    infoNode.append_attribute("Name") = "Typeinfo";
    infoNode.append_attribute("Value") = dataFormToDescription(channel.dataForm).c_str();
    infoNode.append_attribute("Datatype") = typeDescriptorToString(channel.type).c_str();
    infoNode.append_attribute("RequireShift") =
        channel.needShift == Channel::NeedShift::True ? "True" : "False";
            
    // Add one more dimension: number of reals per data item
    auto globalSize = grid->getGridDims()->getGlobalSize();
    globalSize.push_back(channel.nComponents());
            
    auto dataNode = attrNode.append_child("DataItem");
    dataNode.append_attribute("Dimensions") = mirheo::to_string(globalSize).c_str();
    dataNode.append_attribute("NumberType") = numberTypeToString(channel.numberType).c_str();
    dataNode.append_attribute("Precision") = std::to_string(numberTypeToPrecision(channel.numberType)).c_str();
    dataNode.append_attribute("Format") = "HDF";
    dataNode.text() = (h5filename + ":/" + channel.name).c_str();
}
        
void writeData(pugi::xml_node node, const std::string& h5filename, const Grid *grid, const std::vector<Channel>& channels)
{
    for (auto& channel : channels) 
        writeDataSet(node, h5filename, grid, channel);
}

static bool isMasterRank(MPI_Comm comm)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return (rank == 0);
}
        
void write(const std::string& filename, const std::string& h5filename, MPI_Comm comm, const Grid *grid, const std::vector<Channel>& channels, MirState::TimeType time)
{
    if (isMasterRank(comm)) {
        pugi::xml_document doc;
        auto root = doc.append_child("Xdmf");
        root.append_attribute("Version") = "3.0";
        auto domain = root.append_child("Domain");

        auto gridNode = grid->writeToXMF(domain, h5filename);
                
        if (time > -1e-6) gridNode.append_child("Time").append_attribute("Value") = std::to_string(time).c_str();
        writeData(gridNode, h5filename, grid, channels);
                
        doc.save_file(filename.c_str());
    }
            
    MPI_Check( MPI_Barrier(comm) );
}

inline Channel::NeedShift getNeedShiftValue(const std::string& str)
{
    if      (str == "True" ) return Channel::NeedShift::True;
    else if (str == "False") return Channel::NeedShift::False;

    die("Unrecognised needShift value: '%s'", str.c_str());
    return Channel::NeedShift::False;
}

static Channel readDataSet(pugi::xml_node node)
{
    auto infoNode = node.child("Information");
    auto dataNode = node.child("DataItem");

    std::string name            = node.attribute("Name").value();
    std::string formDescription = infoNode.attribute("Value").value();
    std::string typeDescription = infoNode.attribute("Datatype").value();
    std::string needShiftStr    = infoNode.attribute("RequireShift").value();

    auto dataForm = descriptionToDataForm(formDescription);
    auto dataType = stringToTypeDescriptor(typeDescription);
    auto needShift = getNeedShiftValue(needShiftStr);

    std::string channelNumberType = dataNode.attribute("NumberType").value();
    int precision = dataNode.attribute("Precision").as_int();
    auto numberType = infoToNumberType(channelNumberType, precision);

    if (dataForm == Channel::DataForm::Other)
        die("Unrecognised form %s", formDescription.c_str());
            
    return Channel(name, nullptr, dataForm, numberType, dataType, needShift);
}
        
static void readData(pugi::xml_node node, std::vector<Channel>& channels)
{
    for (auto attr : node.children("Attribute"))
        if (std::string(attr.name()) == "Attribute")
            channels.push_back(readDataSet(attr));
}

std::tuple<std::string /*h5filename*/, std::vector<Channel>>
read(const std::string& filename, MPI_Comm comm, Grid *grid)
{
    debug2("Reading XMF header");

    std::string h5filename;
    std::vector<Channel> channels;
    
    pugi::xml_document doc;
    auto parseResult = doc.load_file(filename.c_str());

    if (!parseResult)
        die("parsing error while reading '%s'.\n"
            "\tError description: %s", filename.c_str(), parseResult.description());

    auto gridNode = doc.child("Xdmf").child("Domain").child("Grid");

    grid->readFromXMF(gridNode, h5filename);

    readData(gridNode, channels);

    MPI_Check( MPI_Barrier(comm) );

    debug2("Done with the XMF header");

    return {std::move(h5filename), std::move(channels)};
}

} // namespace XMF
} // namespace XDMF

} // namespace mirheo
