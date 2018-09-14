#include "hdf5_helpers.h"

#include <core/logger.h>

namespace XDMF
{
    namespace HDF5
    {
        hid_t create(std::string filename, MPI_Comm comm)
        {
            hid_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(plist_id_access, comm, MPI_INFO_NULL);  // TODO: add smth here to speed shit up

            hid_t file_id = H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access );

            H5Pclose(plist_id_access);
            
            return file_id;
        }
        
        void writeDataSet(hid_t file_id, const Grid* grid, const Channel& channel)
        {
            debug2("Writing channel '%s' with %d / %d elements of %d floats each",
                   channel.name.c_str(), grid->getLocalSize(), grid->getGlobalSize(), channel.entrySize_floats);
            
            // Add one more dimension: number of floats per data item
            int ndims = grid->getDims() + 1;
            auto localSize = grid->getLocalSize();
            auto globalSize = grid->getGlobalSize();
            
            // What. The. F.
            std::reverse(localSize .begin(), localSize .end());
            std::reverse(globalSize.begin(), globalSize.end());
            
            localSize.push_back(channel.entrySize_floats);
            globalSize.push_back(channel.entrySize_floats);
            
            hid_t filespace_simple = H5Screate_simple(ndims, globalSize.data(), nullptr);

            hid_t dset_id = H5Dcreate(file_id, channel.name.c_str(), H5T_NATIVE_FLOAT, filespace_simple, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);

            H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

            hid_t dspace_id = H5Dget_space(dset_id);

            // TODO check if this is needed
            if (!grid->localEmpty())
                H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, grid->getOffsets().data(), nullptr, localSize.data(), nullptr);
            else
                H5Sselect_none(dspace_id);

            hid_t mspace_id = H5Screate_simple(ndims, localSize.data(), nullptr);

            H5Dwrite(dset_id, H5T_NATIVE_FLOAT, mspace_id, dspace_id, xfer_plist_id, channel.data);

            H5Sclose(mspace_id);
            H5Sclose(dspace_id);
            H5Pclose(xfer_plist_id);
            H5Dclose(dset_id);
        }
        
        void writeData(hid_t file_id, const Grid* grid, const std::vector<Channel>& channels)
        {
            for (auto& channel : channels) 
                writeDataSet(file_id, grid, channel);
        }
            
        void close(hid_t file_id)
        {
            H5Fclose(file_id);
        }
        
        void write(std::string filename, MPI_Comm comm, const Grid* grid, const std::vector<Channel>& channels)
        {
            auto file_id = create(filename, comm);
            if (file_id < 0)
            {
                if (file_id < 0) error("HDF5 failed to write to file '%s'", filename.c_str());
                return;
            }
            
            grid->write2HDF5(file_id, comm);
            writeData(file_id, grid, channels);
            
            close(file_id);
        }
    }
}
