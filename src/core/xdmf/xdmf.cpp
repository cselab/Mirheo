#include "xdmf.h"
#include "common.h"

#include "xmf_helpers.h"
#include "hdf5_helpers.h"

#include <core/logger.h>
#include <core/rigid_kernels/rigid_motion.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/utils/timer.h>

#include <hdf5.h>

namespace XDMF
{
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, float time, MPI_Comm comm)
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
    
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, MPI_Comm comm)
{
    constexpr float arbitraryTime = -1.f;
    write(filename, grid, channels, arbitraryTime, comm);
}

inline long getLocalNumElements(const GridDims *gridDims)
{
    long n = 1;
    for (auto i : gridDims->getLocalSize())  n *= i;
    return n;
}

VertexChannelsData readVertexData(const std::string& filename, MPI_Comm comm, int chunkSize)
{
    info("Reading XDMF vertex data from %s", filename.c_str());

    std::string h5filename;
    VertexChannelsData vertexData;

    auto positions = std::make_shared<std::vector<float3>>();
    VertexGrid grid(positions, comm);

    mTimer timer;
    timer.start();
    std::tie(h5filename, vertexData.descriptions) = XMF::read(filename, comm, &grid);
    grid.splitReadAccess(comm, chunkSize);

    h5filename = makePath(parentPath(filename)) + h5filename;

    long nElements = getLocalNumElements(grid.getGridDims());
    int  nChannels = vertexData.descriptions.size();
    
    vertexData.data.resize(nChannels);

    debug("Got %d channels with %ld items each", nChannels, nElements);

    for (int i = 0; i < nChannels; ++i)
    {
        auto& data = vertexData.data[i];
        auto& desc = vertexData.descriptions[i];
        
        auto sz = nElements * desc.nComponents() * desc.precision();
        data.resize(sz);
        desc.data = data.data();
    }
        
    HDF5::read(h5filename, comm, &grid, vertexData.descriptions);
    info("Reading took %f ms", timer.elapsed());

    vertexData.positions = std::move(*positions);
    return vertexData;
}

} // namespace XDMF
