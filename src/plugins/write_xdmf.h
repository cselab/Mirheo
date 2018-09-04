#pragma once

#include <mpi.h>
#include <string>
#include <vector>
#include <hdf5.h>

class XDMFDumper 
{
public:
    enum class ChannelType : int {Scalar, Vector, Tensor6};

protected:

    struct ChannelInfo
    {
        std::string type;
        int dims;
    };
    
    MPI_Comm xdmfComm;
    std::string path, fname;
    std::vector<std::string> channelNames;

    bool activated{true};
    int timeStamp{0};

    const int zeroPadding = 5;

    int nranks[3], periods[3], my3Drank[3];
    int myrank;

    std::vector<ChannelType> channelTypes;

protected:
    
    virtual void writeXMFHeader   (FILE *xmf, float t) = 0;
    virtual void writeXMFFooter   (FILE *xmf) = 0;
    virtual void writeXMFGeometry (FILE *xmf, std::string currentFname) = 0;
    virtual void writeXMFData     (FILE *xmf, std::string currentFname) = 0;
    
    void writeLight(std::string fname, float t);

    hid_t createIOFile(std::string filename) const;
    void closeIOFile(hid_t file_id) const;

    void writeDataSet(hid_t file_id, int rank, hsize_t globalSize[], hsize_t localSize[], hsize_t offset[],
                      std::string channelName, const float *channelData) const;
    
    std::string getFilename();

    ChannelInfo getInfoFromType(ChannelType type) const;
    
public:
    XDMFDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix,
               std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes);

    ~XDMFDumper();
};
