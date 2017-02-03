#pragma once

#include <mpi.h>
#include <string>
#include <vector>


class XDMFDumper
{
public:
	enum ChannelType {Scalar, Vector};

private:
    MPI_Comm xdmfComm;
    std::string path;
    std::vector<std::string> channelNames;
    int3 dimensions;
    float3 h;
    bool deactivated;
    int timeStamp;

    const int zeroPadding = 5;

	int nranks[3], periods[3], my3Drank[3];
	int myrank;

	std::vector<ChannelType> channelTypes;

    void writeLight(std::string fname, float t);
    void writeHeavy(std::string fname, std::vector<const float*> channelData);

public:
    XDMFDumper(MPI_Comm comm, int3 nranks3D, std::string path, int3 dimensions, float3 h,
    		std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes);

    void dump(std::vector<const float*> channelData, const float t);

    ~XDMFDumper();
};
