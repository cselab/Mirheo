#include <core/logger.h>

#include <hdf5.h>
#include <regex>
#include <string>

#include "timer.h"
#include "write_xdmf_grid.h"

void XDMFGridDumper::writeXMFHeader(FILE *xmf, float t)
{
    fprintf(xmf, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
    fprintf(xmf, "     <Time Value=\"%.f\"/>\n", t);
}

void XDMFGridDumper::writeXMFFooter(FILE *xmf)
{
    fprintf(xmf, "   </Grid>\n");
}

void XDMFGridDumper::writeXMFGeometry(FILE *xmf, std::string currentFname)
{
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
}

void XDMFGridDumper::writeXMFData(FILE *xmf, std::string currentFname)
{
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
    XDMFDumper(comm, nranks3D, fileNamePrefix, channelNames, channelTypes),
    localResolution(localResolution), h(h)
{
    globalResolution.x = nranks3D.x * localResolution.x;
    globalResolution.y = nranks3D.y * localResolution.y;
    globalResolution.z = nranks3D.z * localResolution.z;
}

void XDMFGridDumper::dump(std::vector<const float*> channelData, const float t)
{
    if (!activated) return;

    std::string currentFname = getFilename();
    
    Timer<> timer;
    timer.start();
    if (myrank == 0) this->writeLight(currentFname, t);
    this->writeHeavy(path + currentFname, channelData);

    info("XDMF: grid written to: %s in %f ms", (path + currentFname+"[.h5 .xmf]").c_str(), timer.elapsed());
}
