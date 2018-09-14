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
    SimpleSerializer::deserialize(data, nranks3D, rank3D, resolution, h, sizes);
    
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
    int periods[] = {0, 0, 0};
    MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &cartComm) );
    grid = std::make_unique<XDMF::UniformGrid>(resolution, nranks3D*resolution, h, cartComm);
    
    // TODO: implement serialization of vector of strings
    
    auto init_channel = [] (XDMF::Channel::Type type, int sz) {
        return XDMF::Channel("", nullptr, type, sz*sizeof(float), "float" + std::to_string(sz));
    };
    
    for (auto s : sizes)
    {
        switch (s)
        {
            case 1: channels.push_back(init_channel(XDMF::Channel::Type::Scalar,  s)); break;
            case 3: channels.push_back(init_channel(XDMF::Channel::Type::Vector,  s)); break;
            case 6: channels.push_back(init_channel(XDMF::Channel::Type::Tensor6, s)); break;

            default:
                die("Plugin '%s' got %d as a channel size, expected 1, 3 or 6", name.c_str(), s);
        }
    }

    std::string allNames;
    std::string nm;
    
    int shift = SimpleSerializer::totSize(nranks3D, rank3D, resolution, h, sizes);
    for (auto& channel : channels)
    {
        SimpleSerializer::deserialize(data.data() + shift, channel.name);
        shift += SimpleSerializer::totSize(channel.name);
        allNames += channel.name + ", ";
    }

    if (allNames.length() >= 2)
    {
        allNames.pop_back();
        allNames.pop_back();
    }
    
    // Create the required folder
    createFoldersCollective(comm, parentPath(path));

    debug2("Plugin %s was set up to dump channels %s, resolution is %dx%dx%d, path is %s", name.c_str(),
            allNames.c_str(), resolution.x, resolution.y, resolution.z, path.c_str());
}

void UniformCartesianDumper::deserialize(MPI_Status& stat)
{
    float t;
    int totSize = 0;
    SimpleSerializer::deserialize(data, t);
    totSize += SimpleSerializer::totSize(t);

    debug2("Plugin '%s' will dump right now: simulation time %f, time stamp %d",
           name.c_str(), t, timeStamp);

    containers.resize(channels.size());
            
    for (int i=0; i < channels.size(); i++)
    {
        SimpleSerializer::deserialize(data.data() + totSize, containers[i]);
        totSize += SimpleSerializer::totSize(containers[i]);
        channels[i].data = containers[i].data();

        debug3("Received %d bytes, corresponding to the channel '%s'",
                SimpleSerializer::totSize(containers[i]),
                channels[i].name.c_str());
    }

    std::string tstr = std::to_string(timeStamp++);
    std::string fname = path + std::string(zeroPadding - tstr.length(), '0') + tstr;
        
    XDMF::write(fname, grid.get(), channels, t, cartComm);
}

