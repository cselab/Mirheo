#include <core/logger.h>

#include <hdf5.h>
#include <regex>
#include <string>

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

std::string XDMFDumper::getFilename()
{
    std::string tstr = std::to_string(timeStamp++);
    return fname + std::string(zeroPadding - tstr.length(), '0') + tstr;
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
