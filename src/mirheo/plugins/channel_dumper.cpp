#include "channel_dumper.h"
#include "utils/simple_serializer.h"

#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/xdmf/type_map.h>

#include <string>
#include <memory>

namespace mirheo
{

UniformCartesianDumper::UniformCartesianDumper(std::string name, std::string path) :
    PostprocessPlugin(name),
    path(path)
{}

UniformCartesianDumper::~UniformCartesianDumper()
{
    if (cartComm != MPI_COMM_NULL)
        MPI_Check( MPI_Comm_free(&cartComm) );
}

void UniformCartesianDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();
    
    int3 nranks3D, rank3D;
    int3 resolution;
    real3 h;
    std::vector<int> sizes;
    std::vector<std::string> names;
    std::string numberDensityChannelName;
    SimpleSerializer::deserialize(data, nranks3D, rank3D, resolution, h, sizes, names, numberDensityChannelName);
        
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
    int periods[] = {0, 0, 0};
    MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
    grid = std::make_unique<XDMF::UniformGrid>(resolution, h, cartComm);
        
    auto init_channel = [] (XDMF::Channel::DataForm dataForm, const std::string& str)
    {
        return XDMF::Channel(str, nullptr, dataForm, XDMF::getNumberType<real>(),
                             DataTypeWrapper<real>(), XDMF::Channel::NeedShift::False);
    };
    
    // Density is a special channel which is always present
    std::string allNames = numberDensityChannelName;
    channels.push_back(init_channel(XDMF::Channel::DataForm::Scalar, numberDensityChannelName));
    
    for (size_t i = 0; i < sizes.size(); ++i)
    {
        allNames += ", " + names[i];
        switch (sizes[i])
        {
            case 1: channels.push_back(init_channel(XDMF::Channel::DataForm::Scalar,  names[i])); break;
            case 3: channels.push_back(init_channel(XDMF::Channel::DataForm::Vector,  names[i])); break;
            case 6: channels.push_back(init_channel(XDMF::Channel::DataForm::Tensor6, names[i])); break;

            default:
                die("Plugin '%s' got %d as a channel '%s' size, expected 1, 3 or 6", getCName(), sizes[i], names[i].c_str());
        }
    }
    
    // Create the required folder
    createFoldersCollective(comm, parentPath(path));

    debug2("Plugin %s was set up to dump channels %s. Resolution is %dx%dx%d, path is %s", getCName(),
            allNames.c_str(), resolution.x, resolution.y, resolution.z, path.c_str());
}

static void convert(const std::vector<double> &src, std::vector<real> &dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = static_cast<real>(src[i]);
}

void UniformCartesianDumper::deserialize()
{
    MirState::TimeType t;
    MirState::StepType timeStamp;
    SimpleSerializer::deserialize(data, t, timeStamp, recvNumberDnsity, recvContainers);
    
    debug2("Plugin '%s' will dump right now: simulation time %f, time stamp %d",
           getCName(), t, timeStamp);

    convert(recvNumberDnsity, numberDnsity);    
    channels[0].data = numberDnsity.data();

    containers.resize(recvContainers.size());
    
    for (size_t i = 0; i < recvContainers.size(); ++i)
    {
        convert(recvContainers[i], containers[i]);
        channels[i+1].data = containers[i].data();
    }

    const std::string fname = path + getStrZeroPadded(timeStamp, zeroPadding);
    XDMF::write(fname, grid.get(), channels, t, cartComm);
}

XDMF::Channel UniformCartesianDumper::getChannelOrDie(std::string chname) const
{
    for (const auto& ch : channels)
        if (ch.name == chname)
            return ch;
        
    die("No such channel in plugin '%s' : '%s'", getCName(), chname.c_str());
   
   // Silence the noreturn warning
   return channels[0];
}

std::vector<int> UniformCartesianDumper::getLocalResolution() const
{
    std::vector<int> res;
    for (auto v : grid->getGridDims()->getLocalSize())
        res.push_back(static_cast<int>(v));
    
    return res;
}

} // namespace mirheo
