#pragma once

#include <mpi.h>
#include <string>
#include <vector>


class XDMFDumper
{
private:
    MPI_Comm xdmfcomm;
    std::string path;
    std::vector<std::string> channelNames;
    int3 dimensions;
    float3 h;
    bool deactivated;
    int timeStamp;

    const int padding = 5;

	int nranks[3], periods[3], my3Drank[3];
	int myrank;

    void writeLight(std::string fname, float t);
    void writeHeavy(std::string fname, std::vector<float*> channelData);

public:
    XDMFDumper(MPI_Comm comm, std::string path, int3 dimensions, float3 h, std::vector<std::string> channelNames);

    void dump(std::vector<float*> channelData, const float t);

    ~XDMFDumper();
};
