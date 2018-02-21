#pragma once

#include <mpi.h>
#include <string>
#include <vector>

class XDMFDumper
{
public:
	enum class ChannelType : int {Scalar, Vector, Tensor6};

private:
    MPI_Comm xdmfComm;
    std::string path, fname;
    std::vector<std::string> channelNames;
    int3 localResolution, globalResolution;
    float3 h;
    bool activated{true};
    int timeStamp{0};

    const int zeroPadding = 5;

	int nranks[3], periods[3], my3Drank[3];
	int myrank;

	std::vector<ChannelType> channelTypes;

    void writeLight(std::string fname, float t);
    void writeHeavy(std::string fname, std::vector<const float*> channelData);

public:
    XDMFDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix, int3 localResolution, float3 h,
    		std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes);

    void dump(std::vector<const float*> channelData, const float t);

    ~XDMFDumper();
};
