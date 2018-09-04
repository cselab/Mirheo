#pragma once

#include "write_xdmf.h"

class XDMFParticlesDumper : public XDMFDumper
{
protected:
    void writeXMFHeader   (FILE *xmf, float t) override;
    void writeXMFFooter   (FILE *xmf) override;
    void writeXMFGeometry (FILE *xmf, std::string currentFname) override;
    void writeXMFData     (FILE *xmf, std::string currentFname) override;

    void writeHeavy(std::string currentFname, int nparticles, const float *positions, std::vector<const float*> channelData);

    long num_particles_tot {0};
    
public:
    XDMFParticlesDumper(MPI_Comm comm, int3 nranks3D, std::string fileNamePrefix,
                        std::vector<std::string> channelNames, std::vector<ChannelType> channelTypes);

    ~XDMFParticlesDumper();

    void dump(int nparticles, const float *positions, std::vector<const float*> channelData, const float t);
};
