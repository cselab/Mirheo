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
}
