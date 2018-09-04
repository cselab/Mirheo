#pragma once

#include <mpi.h>
#include <string>
#include <vector>

class XDMFDumper 
{
public:
    enum class ChannelType : int {Scalar, Vector, Tensor6};

protected:
    MPI_Comm xdmfComm;
    std::string path, fname;
    std::vector<std::string> channelNames;

    bool activated{true};
    int timeStamp{0};

    const int zeroPadding = 5;

    int nranks[3], periods[3], my3Drank[3];
    int myrank;

    std::vector<ChannelType> channelTypes;

    virtual void writeXMFHeader   (FILE *xmf, float t) = 0;
    virtual void writeXMFFooter   (FILE *xmf) = 0;
    virtual void writeXMFGeometry (FILE *xmf, std::string currentFname) = 0;
    virtual void writeXMFData     (FILE *xmf, std::string currentFname) = 0;
    
    void writeLight(std::string fname, float t);
    virtual void writeHeavy(std::string fname, std::vector<const float*> channelData) = 0;

    std::string getFilename();
    
public:
    XDMFDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix,
               std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes);

    ~XDMFDumper();
};
