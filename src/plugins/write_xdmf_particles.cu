#include <core/logger.h>

#include <hdf5.h>
#include <string>

#include "timer.h"
#include "write_xdmf_particles.h"

static const char positionChanelName[] = "positions";

void XDMFParticlesDumper::writeXMFHeader(FILE *xmf, float t)
{
    fprintf(xmf, "   <Grid Name=\"particles\"\n");
    fprintf(xmf, "     <Time Value=\"%.f\"/>\n", t);
}

void XDMFParticlesDumper::writeXMFFooter(FILE *xmf)
{
    fprintf(xmf, "   </Grid>\n");
}

void XDMFParticlesDumper::writeXMFGeometry(FILE *xmf, std::string currentFname)
{
    fprintf(xmf, "     <Topology TopologyType=\"Polyvertex\" NodesPerElement=\"%d\"/>\n",
            num_particles_tot);

    fprintf(xmf, "     <Geometry GeometryType=\"XYZ\">\n");
    fprintf(xmf, "       <DataItem DataType=\"Float\" Dimensions=\"%d 3\" Format=\"HDF\">\n");
    fprintf(xmf, "        %s:/%s\n", (currentFname+".h5").c_str(), positionChanelName);
    fprintf(xmf, "       </DataItem>\n");
    fprintf(xmf, "     </Geometry>\n");
}

void XDMFParticlesDumper::writeXMFData(FILE *xmf, std::string currentFname)
{
    for(int ichannel = 0; ichannel < channelNames.size(); ichannel++) {

        auto info = getInfoFromType(channelTypes[ichannel]);

        fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"%s\" Center=\"Node\">\n", channelNames[ichannel].c_str(), info.type.c_str());
        fprintf(xmf, "       <DataItem Dimensions=\"%d %d\" DataType=\"Float\" Format=\"HDF\">\n", num_particles_tot, info.dims);

        fprintf(xmf, "        %s:/%s\n", (currentFname+".h5").c_str(), channelNames[ichannel].c_str());

        fprintf(xmf, "       </DataItem>\n");
        fprintf(xmf, "     </Attribute>\n");
    }
}


void XDMFParticlesDumper::writeHeavy(std::string currentFname, int nparticles, const float *positions, std::vector<const float*> channelData)
{
    // TODO
}


XDMFParticlesDumper::XDMFParticlesDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix,
                                         std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes) :
    XDMFDumper(comm, nranks3D, fileNamePrefix, channelNames, channelTypes)
{}

void XDMFParticlesDumper::dump(int nparticles, const float *positions, std::vector<const float*> channelData, const float t)
{
    if (!activated) return;

    int num_particles_loc = nparticles;

    MPI_Check( MPI_Allreduce(&num_particles_loc, &num_particles_tot, 1, MPI_INT, MPI_SUM, xdmfComm) );

    std::string currentFname = getFilename();
    
    Timer<> timer;
    timer.start();
    if (myrank == 0) this->writeLight(currentFname, t);
    this->writeHeavy(path + currentFname, nparticles, positions, channelData);

    info("XDMF: particles written to: %s in %f ms", (path + currentFname+"[.h5 .xmf]").c_str(), timer.elapsed());
    
}
