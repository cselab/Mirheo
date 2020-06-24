// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "xdmf.h"
#include "common.h"

#include "xmf_helpers.h"
#include "hdf5_helpers.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/folders.h>
#include <mirheo/core/utils/timer.h>

#include <hdf5.h>

namespace mirheo
{

namespace XDMF
{
void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, MirState::TimeType time, MPI_Comm comm)
{
    std::string h5Filename  = filename + ".h5";
    std::string xmfFilename = filename + ".xmf";

    info("Writing XDMF data to %s[.h5,.xmf]", filename.c_str());

    mTimer timer;
    timer.start();
    XMF::write(xmfFilename, getBaseName(h5Filename), comm, grid, channels, time);
    HDF5::write(h5Filename, comm, grid, channels);
    info("Writing took %f ms", timer.elapsed());
}

void write(const std::string& filename, const Grid *grid,
           const std::vector<Channel>& channels, MPI_Comm comm)
{
    constexpr MirState::TimeType arbitraryTime = -1.0;
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

    auto positions = std::make_shared<std::vector<real3>>();
    VertexGrid grid(positions, comm);

    mTimer timer;
    timer.start();
    std::tie(h5filename, vertexData.descriptions) = XMF::read(filename, comm, &grid);
    grid.splitReadAccess(comm, chunkSize);

    h5filename = joinPaths(getParentPath(filename), h5filename);

    const size_t nElements = getLocalNumElements(grid.getGridDims());
    const size_t  nChannels = vertexData.descriptions.size();

    vertexData.data.resize(nChannels);

    debug("Got %lud channels with %ld items each", nChannels, nElements);

    for (size_t i = 0; i < nChannels; ++i)
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

} // namespace mirheo
