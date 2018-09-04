#include <core/logger.h>

#include <hdf5.h>
#include <regex>
#include <string>

#include "timer.h"
#include "write_xdmf.h"


void XDMFGridDumper::writeLight(std::string currentFname, float t)
{
    FILE* xmf;
    xmf = fopen( (path+currentFname+".xmf").c_str(), "w" );
    if (xmf == nullptr)
    {
        if (myrank == 0) error("XMF write failed: %s", (fname+".xmf").c_str());
        return;
    }

    fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
    fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
    fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
    fprintf(xmf, " <Domain>\n");
    fprintf(xmf, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(xmf, "     <Time Value=\"%.f\"/>\n", t);

    // WTF resolution should go in Z-Y-X order! Achtung aliens attack!!
    fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n",
            globalResolution.z+1, globalResolution.y+1, globalResolution.x+1);

    fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
    fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
    fprintf(xmf, "        %e %e %e\n", 0.0, 0.0, 0.0);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");

    fprintf(xmf, "        %e %e %e\n", h.x, h.y, h.z);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n");

    for(int ichannel = 0; ichannel < channelNames.size(); ichannel++)
    {
        std::string type;
        int dims;
        switch (channelTypes[ichannel])
        {
            case ChannelType::Scalar:  type = "Scalar";  dims = 1;  break;
            case ChannelType::Vector:  type = "Vector";  dims = 3;  break;
            case ChannelType::Tensor6: type = "Tensor6"; dims = 6;  break;
        }

        fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"%s\" Center=\"Cell\">\n", channelNames[ichannel].c_str(), type.c_str());
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n",
                globalResolution.x, globalResolution.y, globalResolution.z, dims);

        fprintf(xmf, "        %s:/%s\n", (currentFname+".h5").c_str(), channelNames[ichannel].c_str());

        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
    }

    fprintf(xmf, "   </Grid>\n");
    fprintf(xmf, " </Domain>\n");
    fprintf(xmf, "</Xdmf>\n");

    fclose(xmf);
}

void XDMFGridDumper::writeHeavy(std::string currentFname, std::vector<const float*> channelData)
{
    hid_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id_access, xdmfComm, MPI_INFO_NULL);  // TODO: add smth here to speed shit up

    hid_t file_id = H5Fcreate( (currentFname+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access );
    if (file_id < 0)
    {
        error("HDF5 write failed: %s", (currentFname+".h5").c_str());
        return;
    }

    H5Pclose(plist_id_access);

    for(int ichannel = 0; ichannel < channelNames.size(); ++ichannel)
    {
        hsize_t dims;
        switch (channelTypes[ichannel])
        {
            case ChannelType::Scalar:  dims = 1;  break;
            case ChannelType::Vector:  dims = 3;  break;
            case ChannelType::Tensor6: dims = 6;  break;
        }

        hsize_t globalsize[4] = { (hsize_t)globalResolution.z,
                                  (hsize_t)globalResolution.y,
                                  (hsize_t)globalResolution.x,  dims};
        hid_t filespace_simple = H5Screate_simple(4, globalsize, nullptr);

        hid_t dset_id = H5Dcreate(file_id, channelNames[ichannel].c_str(), H5T_NATIVE_FLOAT, filespace_simple, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);

        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        hsize_t start[4] = { (hsize_t)my3Drank[2] * localResolution.z,
                             (hsize_t)my3Drank[1] * localResolution.y,
                             (hsize_t)my3Drank[0] * localResolution.x, (hsize_t)0 };

        hsize_t extent[4] = { (hsize_t)localResolution.z, (hsize_t)localResolution.y, (hsize_t)localResolution.x, dims };
        hid_t filespace = H5Dget_space(dset_id);
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);

        hid_t memspace = H5Screate_simple(4, extent, NULL);
        herr_t status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, channelData[ichannel]);

        H5Sclose(memspace);
        H5Sclose(filespace);
        H5Pclose(plist_id);
        H5Dclose(dset_id);
    }

    H5Fclose(file_id);
}

XDMFGridDumper::XDMFGridDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix, int3 localResolution, float3 h,
                               std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes) :
    localResolution(localResolution), h(h),
    channelNames(channelNames), channelTypes(channelTypes)
{
    int ranksArr[] = {nranks3D.x, nranks3D.y, nranks3D.z};
    globalResolution.x = nranks3D.x * localResolution.x;
    globalResolution.y = nranks3D.y * localResolution.y;
    globalResolution.z = nranks3D.z * localResolution.z;

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

void XDMFGridDumper::dump(std::vector<const float*> channelData, const float t)
{
    if (!activated) return;

    std::string tstr = std::to_string(timeStamp++);
    std::string currentFname = fname + std::string(zeroPadding - tstr.length(), '0') + tstr;

    Timer<> timer;
    timer.start();
    if (myrank == 0) writeLight(currentFname, t);
    writeHeavy(path + currentFname, channelData);

    info("XDMF written to: %s in %f ms", (path + currentFname+"[.h5 .xmf]").c_str(), timer.elapsed());
}
