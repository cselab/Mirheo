#include "xdmf.h"
#include "common.h"

#include "xmf_helpers.h"
#include "hdf5_helpers.h"

#include <hdf5.h>

#include <core/logger.h>
#include <core/utils/timer.h>
#include <core/utils/folders.h>

namespace XDMF
{
    void write(std::string filename, const Grid* grid, const std::vector<Channel>& channels, float time, MPI_Comm comm)
    {        
        std::string h5Filename  = filename + ".h5";
        std::string xmfFilename = filename + ".xmf";
        
        info("Writing XDMF data to %s[.h5,.xmf]", filename.c_str());

        mTimer timer;
        timer.start();
        XMF::write(xmfFilename, relativePath(h5Filename), comm, grid, channels, time);
        HDF5::write(h5Filename, comm, grid, channels);
        info("Writing took %f ms", timer.elapsed());
    }
    
    void write(std::string filename, const Grid* grid, const std::vector<Channel>& channels, MPI_Comm comm)
    {        
        write(filename, grid, channels, -1, comm);
    }

    static long getLocalNumElements(const Grid *grid)
    {
        long n = 1;
        for (auto i : grid->getLocalSize())  n *= i;
        return n;
    }

    static void gatherChannels(std::vector<Channel> &channels, std::vector<float> &positions, ParticleVector *pv)
    {
        int n = positions.size() / 3;
        const float3 *pos, *vel = nullptr;
        const int *ids = nullptr;

        pv->local()->resize_anew(n);
        auto& coosvels = pv->local()->coosvels;

        for (auto& ch : channels)
        {
            if (ch.name == "velocity")
                vel = (const float3*) ch.data;
            
            if (ch.name == "ids")
                ids = (const int*) ch.data;
        }

        if (n > 0 && vel == nullptr)
            die("Channel 'velocities' is required to read XDMF into a particle vector");
        if (n > 0 && ids == nullptr)
            die("Channel 'ids' is required to read XDMF into a particle vector");

        pos = (const float3*) positions.data();
        
        for (int i = 0; i < n; ++i)
        {
            Particle p;
            p.r = pos[i];
            p.u = vel[i];
            p.i1 = ids[i];
            
            coosvels.hostPtr()[i] = p;
        }

        coosvels.uploadToDevice(0);

        // TODO extra data
    }
    
    void readParticleData(std::string filename, MPI_Comm comm, ParticleVector *pv, int chunk_size)
    {
        info("Reading XDMF data from %s", filename.c_str());

        std::string h5filename;
        
        auto positions = std::make_shared<std::vector<float>>();
        std::vector<std::vector<char>> channelData;
        std::vector<Channel> channels;
        
        VertexGrid grid(positions, comm);

        mTimer timer;
        timer.start();
        XMF::read(filename, comm, h5filename, &grid, channels);
        grid.split_read_access(comm, chunk_size);

        h5filename = parentPath(filename) + h5filename;

        long nElements = getLocalNumElements(&grid);
        channelData.resize(channels.size());        

        for (int i = 0; i < channels.size(); ++i) {
            channelData[i].resize(nElements * channels[i].nComponents() * channels[i].precision());
            channels[i].data = channelData[i].data();
        }
        
        HDF5::read(h5filename, comm, &grid, channels);
        info("Reading took %f ms", timer.elapsed());

        gatherChannels(channels, *positions, pv);
    }

    static void gatherChannels(std::vector<Channel> &channels, std::vector<float> &positions, ObjectVector *ov)
    {
        int n = positions.size() / 3;
        const int *ids_data = nullptr;

        auto ids = ov->local()->extraPerObject.getData<int>("ids");

        ids->resize_anew(n);

        for (auto& ch : channels)
        {
            if (ch.name == "ids")
                ids_data = (const int*) ch.data;
        }

        if (n > 0 && ids_data == nullptr)
            die("Channel 'ids' is required to read XDMF into an object vector");

        for (int i = 0; i < n; ++i)
        {
            (*ids)[i] = ids_data[i];
        }

        ids->uploadToDevice(0);

        // TODO extra data
    }

    void readObjectData(std::string filename, MPI_Comm comm, ObjectVector *ov)
    {
        info("Reading XDMF data from %s", filename.c_str());

        std::string h5filename;
        
        auto positions = std::make_shared<std::vector<float>>();
        std::vector<std::vector<char>> channelData;
        std::vector<Channel> channels;
        
        VertexGrid grid(positions, comm);

        mTimer timer;
        timer.start();
        XMF::read(filename, comm, h5filename, &grid, channels);
        grid.split_read_access(comm, 1);

        h5filename = parentPath(filename) + h5filename;

        long nElements = getLocalNumElements(&grid);
        channelData.resize(channels.size());        

        for (int i = 0; i < channels.size(); ++i) {
            channelData[i].resize(nElements * channels[i].nComponents() * channels[i].precision());
            channels[i].data = channelData[i].data();
        }
        
        HDF5::read(h5filename, comm, &grid, channels);
        info("Reading took %f ms", timer.elapsed());

        gatherChannels(channels, *positions, ov);
    }

    void readRigidObjectData(std::string filename, MPI_Comm comm, ObjectVector *ov)
    {
        info("Reading XDMF data from %s", filename.c_str());

        std::string h5filename;
        
        auto positions = std::make_shared<std::vector<float>>();
        std::vector<std::vector<char>> channelData;
        std::vector<Channel> channels;
        
        VertexGrid grid(positions, comm);

        mTimer timer;
        timer.start();
        XMF::read(filename, comm, h5filename, &grid, channels);
        grid.split_read_access(comm, 1);

        h5filename = parentPath(filename) + h5filename;

        long nElements = getLocalNumElements(&grid);
        channelData.resize(channels.size());        

        for (int i = 0; i < channels.size(); ++i) {
            channelData[i].resize(nElements * channels[i].nComponents() * channels[i].precision());
            channels[i].data = channelData[i].data();
        }
        
        HDF5::read(h5filename, comm, &grid, channels);
        info("Reading took %f ms", timer.elapsed());

        gatherChannels(channels, *positions, ov);
    }

}
