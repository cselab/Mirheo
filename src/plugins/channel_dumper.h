#pragma once

#include <plugins/interface.h>
#include <plugins/write_xdmf.h>

#include <vector>
#include <mpi.h>

class UniformCartesianDumper : public PostprocessPlugin
{
public:
    UniformCartesianDumper(std::string name, std::string path);

    void deserialize(MPI_Status& stat) override;
    void handshake() override;

protected:
    XDMFDumper* dumper;
    std::string path;

    int3 nranks3D, rank3D;
    int3 resolution;
    float3 h;

    std::vector<XDMFDumper::ChannelType> channelTypes;
    std::vector<std::string> channelNames;
    std::vector<std::vector<float>> channels;
};
