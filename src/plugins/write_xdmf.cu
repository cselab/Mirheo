#include <regex>

#include <core/logger.h>

#include "timer.h"
#include "write_xdmf.h"

static void write_xdmf_header(FILE *xmf)
{
    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n"); 
}

static void write_xdmf_footer(FILE *xmf)
{
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");
}

void XDMFDumper::writeLight(std::string currentFname, float t)
{
    FILE* xmf;
    xmf = fopen( (path+currentFname+".xmf").c_str(), "w" );
    if (xmf == nullptr)
    {
        if (myrank == 0) error("XMF write failed: %s", (fname+".xmf").c_str());
        return;
    }

    write_xdmf_header(xmf);

    this->writeXMFHeader    (xmf, t);
    this->writeXMFGeometry  (xmf, currentFname);
    this->writeXMFData      (xmf, currentFname);
    this->writeXMFFooter    (xmf);

    write_xdmf_footer(xmf);

    fclose(xmf);
}

hid_t XDMFDumper::createIOFile(std::string filename) const
{
    hid_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id_access, xdmfComm, MPI_INFO_NULL);  // TODO: add smth here to speed shit up

    hid_t file_id = H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access );

    if (file_id < 0) {
        error("HDF5 write failed: %s", filename.c_str());
        return -1;
    }

    H5Pclose(plist_id_access);

    return file_id;
}

void XDMFDumper::closeIOFile(hid_t file_id) const
{
    H5Fclose(file_id);
}

void XDMFDumper::writeDataSet(hid_t file_id, int rank, hsize_t globalSize[], hsize_t localSize[], hsize_t offset[],
                              std::string channelName, const float *channelData) const
{
    int nLocData = 1, nTotData = 1;
    for (int i = 0; i < rank; ++i) {
        nLocData *= localSize[i];
        nTotData *= globalSize[i];
    }
    
    hid_t filespace_simple = H5Screate_simple(rank, globalSize, nullptr);

    hid_t dset_id = H5Dcreate(file_id, channelName.c_str(), H5T_NATIVE_FLOAT, filespace_simple, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);

    H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

    hid_t dspace_id = H5Dget_space(dset_id);

    if (nLocData)
        H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, offset, nullptr, localSize, nullptr);
    else
        H5Sselect_none(dspace_id);

    hid_t mspace_id = H5Screate_simple(rank, localSize, nullptr);

    if (nTotData)
        H5Dwrite(dset_id, H5T_NATIVE_FLOAT, mspace_id, dspace_id, xfer_plist_id, channelData);

    H5Sclose(mspace_id);
    H5Sclose(dspace_id);
    H5Pclose(xfer_plist_id);
    H5Dclose(dset_id);
}


std::string XDMFDumper::getFilename()
{
    std::string tstr = std::to_string(timeStamp++);
    return fname + std::string(zeroPadding - tstr.length(), '0') + tstr;
}

XDMFDumper::ChannelInfo XDMFDumper::getInfoFromType(XDMFDumper::ChannelType type) const
{
    ChannelInfo info;
    switch (type) {
    case ChannelType::Scalar:  info.type = "Scalar";  info.dims = 1;  break;
    case ChannelType::Vector:  info.type = "Vector";  info.dims = 3;  break;
    case ChannelType::Tensor6: info.type = "Tensor6"; info.dims = 6;  break;
    }
    return info;
}

XDMFDumper::XDMFDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix,
                       std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes) :
    channelNames(channelNames), channelTypes(channelTypes)
{
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};

    MPI_Check( MPI_Cart_create(comm, 3, ranksArr, periods, 0, &xdmfComm) );
    MPI_Check( MPI_Cart_get(xdmfComm, 3, nranks, periods, my3Drank) );
    MPI_Check( MPI_Comm_rank(xdmfComm, &myrank));

    // Create and setup folders

    std::regex re(R".(^(.*/)(.+)).");
    std::smatch match;
    if (std::regex_match(fileNamePrefix, match, re))
    {
        path  = match[1].str();
        fname = match[2].str();
        std::string command = "mkdir -p " + path;
        if (myrank == 0)
        {
            if ( system(command.c_str()) != 0 )
            {
                error("Could not create folders or files by given path, dumping will be disabled.");
                activated = false;
            }
        }
    }
    else
    {
        path = "";
        fname = fileNamePrefix;
    }
}
