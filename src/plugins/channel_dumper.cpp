#include "channel_dumper.h"
#include "utils/simple_serializer.h"

#include <core/simulation.h>
#include <core/utils/folders.h>

#include <string>
#include <memory>

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
    float3 h;
    std::vector<int> sizes;
    std::vector<std::string> names;
    SimpleSerializer::deserialize(data, nranks3D, rank3D, resolution, h, sizes, names);
        
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
    int periods[] = {0, 0, 0};
    MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
    grid = std::make_unique<XDMF::UniformGrid>(resolution, h, cartComm);
        
    auto init_channel = [] (XDMF::Channel::DataForm dataForm, const std::string& str) {
        return XDMF::Channel(str, nullptr, dataForm, XDMF::Channel::NumberType::Float, DataTypeWrapper<float>());
    };
    
    // Density is a special channel which is always present
    std::string allNames = "density";
    channels.push_back(init_channel(XDMF::Channel::DataForm::Scalar, "density"));
    
    for (int i = 0; i < sizes.size(); i++)
    {
        allNames += ", " + names[i];
        switch (sizes[i])
        {
            case 1: channels.push_back(init_channel(XDMF::Channel::DataForm::Scalar,  names[i])); break;
            case 3: channels.push_back(init_channel(XDMF::Channel::DataForm::Vector,  names[i])); break;
            case 6: channels.push_back(init_channel(XDMF::Channel::DataForm::Tensor6, names[i])); break;

            default:
                die("Plugin '%s' got %d as a channel '%s' size, expected 1, 3 or 6", name.c_str(), sizes[i], names[i].c_str());
        }
    }
    
    // Create the required folder
    createFoldersCollective(comm, parentPath(path));

    debug2("Plugin %s was set up to dump channels %s. Resolution is %dx%dx%d, path is %s", name.c_str(),
            allNames.c_str(), resolution.x, resolution.y, resolution.z, path.c_str());
}

static void convert(const std::vector<double> &src, std::vector<float> &dst)
{
    dst.resize(src.size());
    for (int i = 0; i < src.size(); ++i)
        dst[i] = src[i];
}

void UniformCartesianDumper::deserialize(MPI_Status& stat)
{
    YmrState::TimeType t;
    YmrState::StepType timeStamp;
    SimpleSerializer::deserialize(data, t, timeStamp, recv_density, recv_containers);
    
    debug2("Plugin '%s' will dump right now: simulation time %f, time stamp %d",
           name.c_str(), t, timeStamp);

    convert(recv_density, density);    
    channels[0].data = density.data();

    containers.resize(recv_containers.size());
    
    for (int i = 0; i < recv_containers.size(); i++) {
        convert(recv_containers[i], containers[i]);
        channels[i+1].data = containers[i].data();
    }

    std::string fname = path + getStrZeroPadded(timeStamp, zeroPadding);
    XDMF::write(fname, grid.get(), channels, t, cartComm);
}

XDMF::Channel UniformCartesianDumper::getChannelOrDie(std::string chname) const
{
    for (const auto& ch : channels)
        if (ch.name == chname)
            return ch;
        
    die("No such channel in plugin '%s' : '%s'", name.c_str(), chname.c_str());
   
   // Silence the noreturn warning
   return channels[0];
}

std::vector<int> UniformCartesianDumper::getLocalResolution() const
{
    std::vector<int> res;
    for (auto v : grid->getGridDims()->getLocalSize())
        res.push_back(v);
    
    return res;
}

