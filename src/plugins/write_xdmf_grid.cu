#include <regex>

#include <core/logger.h>
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
    for (int ichannel = 0; ichannel < channelNames.size(); ichannel++) {

        auto info = getInfoFromType(channelTypes[ichannel]);

        fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"%s\" Center=\"Cell\">\n", channelNames[ichannel].c_str(), info.type.c_str());
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d %d\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n",
                globalResolution.x, globalResolution.y, globalResolution.z, info.dims);

        fprintf(xmf, "        %s:/%s\n", (currentFname+".h5").c_str(), channelNames[ichannel].c_str());

        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
    }
}


void XDMFGridDumper::writeHeavy(std::string currentFname, std::vector<const float*> channelData)
{
    auto file_id = createIOFile(currentFname + ".h5");

    if (file_id < 0) return;

    for (int ichannel = 0; ichannel < channelNames.size(); ++ichannel) {
        auto info = getInfoFromType(channelTypes[ichannel]);
        
        hsize_t globalSize[4] = { (hsize_t) globalResolution.z, (hsize_t) globalResolution.y, (hsize_t) globalResolution.x, (hsize_t) info.dims};
        hsize_t localSize [4] = { (hsize_t)  localResolution.z, (hsize_t)  localResolution.y, (hsize_t)  localResolution.x, (hsize_t) info.dims};
        
        hsize_t offset[4] = { (hsize_t) my3Drank[2] * localResolution.z,
                              (hsize_t) my3Drank[1] * localResolution.y,
                              (hsize_t) my3Drank[0] * localResolution.x,
                              (hsize_t) 0 };

        writeDataSet(file_id, 4, globalSize, localSize, offset, channelNames[ichannel], channelData[ichannel]);
    }

    closeIOFile(file_id);
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
