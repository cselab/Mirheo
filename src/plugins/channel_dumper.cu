#include <string>

#include "channel_dumper.h"
#include "simple_serializer.h"
#include <core/utils/folders.h>
#include <core/utils/make_unique.h>
#include <core/simulation.h>


UniformCartesianDumper::UniformCartesianDumper(std::string name, std::string path) :
        PostprocessPlugin(name), path(path)
{   }

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
    grid = std::make_unique<XDMF::UniformGrid>(resolution, nranks3D*resolution, h, cartComm);
        
    auto init_channel = [] (XDMF::Channel::Type type, int sz, const std::string& str) {
        return XDMF::Channel(str, nullptr, type, sz*sizeof(float), "float" + std::to_string(sz));
    };
    
    // Density is a special channel which is always present
    std::string allNames = "density";
    channels.push_back(init_channel(XDMF::Channel::Type::Scalar, 1, "density"));
    
    for (int i=0; i<sizes.size(); i++)
    {
        allNames += ", " + names[i];
        switch (sizes[i])
        {
            case 1: channels.push_back(init_channel(XDMF::Channel::Type::Scalar,  sizes[i], names[i])); break;
            case 3: channels.push_back(init_channel(XDMF::Channel::Type::Vector,  sizes[i], names[i])); break;
            case 6: channels.push_back(init_channel(XDMF::Channel::Type::Tensor6, sizes[i], names[i])); break;

            default:
                die("Plugin '%s' got %d as a channel '%s' size, expected 1, 3 or 6", name.c_str(), sizes[i], names[i].c_str());
        }
    }
    
    // Create the required folder
    createFoldersCollective(comm, parentPath(path));

    debug2("Plugin %s was set up to dump channels %s. Resolution is %dx%dx%d, path is %s", name.c_str(),
            allNames.c_str(), resolution.x, resolution.y, resolution.z, path.c_str());
}

void UniformCartesianDumper::deserialize(MPI_Status& stat)
{
    float t;
    SimpleSerializer::deserialize(data, t, density, containers);
    
    debug2("Plugin '%s' will dump right now: simulation time %f, time stamp %d",
           name.c_str(), t, timeStamp);

    channels[0].data = density.data();
    for (int i=0; i < containers.size(); i++)
        channels[i+1].data = containers[i].data();

    std::string tstr = std::to_string(timeStamp++);
    std::string fname = path + std::string(zeroPadding - tstr.length(), '0') + tstr;
        
    XDMF::write(fname, grid.get(), channels, t, cartComm);
}

