#include "channel_dumper.h"
#include "simple_serializer.h"
#include <core/utils/folders.h>

#include <core/simulation.h>
#include <string>


UniformCartesianDumper::UniformCartesianDumper(std::string name, std::string path) :
        PostprocessPlugin(name), path(path)
{ }

void UniformCartesianDumper::handshake()
{
    auto req = waitData();
    MPI_Check( MPI_Wait(&req, MPI_STATUS_IGNORE) );
    recv();

    std::vector<int> sizes;
    SimpleSerializer::deserialize(data, nranks3D, rank3D, resolution, h, sizes);

    for (auto s : sizes)
    {
        switch (s)
        {
            case 1: channelTypes.push_back(XDMFDumper::ChannelType::Scalar);  break;
            case 3: channelTypes.push_back(XDMFDumper::ChannelType::Vector);  break;
            case 6: channelTypes.push_back(XDMFDumper::ChannelType::Tensor6); break;

            default:
                die("Plugin '%s' got %d as a channel size, expected 1, 3 or 6", name.c_str(), s);
        }
    }

    channels.resize(sizes.size());
    channelNames.resize(sizes.size());

    std::string allNames;
    int shift = SimpleSerializer::totSize(nranks3D, rank3D, resolution, h, sizes);
    for (auto& nm : channelNames)
    {
        SimpleSerializer::deserialize(data.data() + shift, nm);
        shift += SimpleSerializer::totSize(nm);
        allNames += nm + ", ";
    }

    if (allNames.length() >= 2)
    {
        allNames.pop_back();
        allNames.pop_back();
    }

    debug2("Plugin %s was set up to dump channels %s, resolution is %dx%dx%d, path is %s", name.c_str(),
            allNames.c_str(), resolution.x, resolution.y, resolution.z, path.c_str());

    dumper = new XDMFGridDumper(comm, nranks3D, path, resolution, h, channelNames, channelTypes);
}

void UniformCartesianDumper::deserialize(MPI_Status& stat)
{
    float t;
    int totSize = 0;
    SimpleSerializer::deserialize(data, t);
    totSize += SimpleSerializer::totSize(t);

    debug2("Plugin %s will dump right now", name.c_str());

    int c = 0;
    for (auto& channel : channels)
    {
        SimpleSerializer::deserialize(data.data() + totSize, channel);
        totSize += SimpleSerializer::totSize(channel);

        debug3("Received %d bytes, corresponding to the channel '%s'",
                SimpleSerializer::totSize(channel),
                channelNames[c].c_str());

        c++;
    }


    std::vector<const float*> chPtrs;
    for (auto& ch : channels) chPtrs.push_back((const float*)ch.data());

    dumper->dump(chPtrs, t);
}

