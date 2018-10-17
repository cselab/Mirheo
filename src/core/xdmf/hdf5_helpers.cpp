#include "hdf5_helpers.h"

#include <core/logger.h>
#include <cstdlib>
#include <cstring>

namespace XDMF
{
    namespace HDF5
    {

        static hid_t createFileAccess(MPI_Comm comm)
        {
            int size;
            MPI_Check( MPI_Comm_size(comm, &size) );
            
            // Don't set the hints if they are already provided by the env variable
            const char* hints = getenv("MPICH_MPIIO_HINTS");
            
            MPI_Info info;
            if (hints == nullptr || strlen(hints) < 1)
            {
                // Collective buffers for mpi i/o
                int cb = 1;
                while (cb*2 <= size) cb *= 2;

                cb = std::min(cb, 128);
                char cbstr[100];
                sprintf(cbstr, "%d", cb);

                MPI_Info_create(&info);
                MPI_Info_set(info, "cb_nodes", cbstr);
                MPI_Info_set(info, "romio_cb_write", "enable");
                MPI_Info_set(info, "romio_cb_read",  "enable");
                //MPI_Info_set(info, "romio_ds_write", "automatic");
                //MPI_Info_set(info, "romio_ds_read",  "automatic");
                MPI_Info_set(info, "striping_factor", cbstr);
                MPI_Info_set(info, "striping_unit", "4194304");
            }
            else
                info = MPI_INFO_NULL;

            hid_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
            H5Pset_fapl_mpio(plist_id_access, comm, info);
            return plist_id_access;
        }
        
        hid_t create(std::string filename, MPI_Comm comm)
        {
            hid_t access_id = createFileAccess(comm);
            hid_t file_id   = H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access_id );
            H5Pclose(access_id);
            
            return file_id;
        }

        hid_t openReadOnly(std::string filename, MPI_Comm comm)
        {
            hid_t access_id = createFileAccess(comm);
            hid_t file_id   = H5Fopen( filename.c_str(), H5F_ACC_RDONLY, access_id );
            H5Pclose(access_id);
            
            return file_id;
        }
        
        void writeDataSet(hid_t file_id, const Grid* grid, const Channel& channel)
        {
            debug2("Writing channel '%s'", channel.name.c_str());
            
            // Add one more dimension: number of floats per data item
            int ndims = grid->getDims() + 1;
            auto localSize = grid->getLocalSize();
            auto globalSize = grid->getGlobalSize();
            
            // What. The. F.
            std::reverse(localSize .begin(), localSize .end());
            std::reverse(globalSize.begin(), globalSize.end());
            
            localSize .push_back(channel.nComponents());
            globalSize.push_back(channel.nComponents());
            
            // Float or Int
            auto datatype = datatypeToHDF5type(channel.datatype);
            
            hid_t filespace_simple = H5Screate_simple(ndims, globalSize.data(), nullptr);

            hid_t dset_id = H5Dcreate(file_id, channel.name.c_str(), datatype, filespace_simple, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);

            H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);

            hid_t dspace_id = H5Dget_space(dset_id);

            // TODO check if this is needed
            if (!grid->localEmpty())
                H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, grid->getOffsets().data(), nullptr, localSize.data(), nullptr);
            else
                H5Sselect_none(dspace_id);

            hid_t mspace_id = H5Screate_simple(ndims, localSize.data(), nullptr);

            if (!grid->globalEmpty())
                H5Dwrite(dset_id, datatype, mspace_id, dspace_id, xfer_plist_id, channel.data);

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
        
        void readDataSet(hid_t file_id, const Grid* grid, Channel& channel)
        {
            debug2("Reading channel '%s'", channel.name.c_str());

            // Add one more dimension: number of floats per data item
            int ndims = grid->getDims() + 1;
            auto localSize  = grid->getLocalSize();
            
            // What. The. F.
            std::reverse(localSize .begin(), localSize .end());
            
            localSize.push_back(channel.nComponents());
            
            hid_t dset_id       = H5Dopen(file_id, channel.name.c_str(), H5P_DEFAULT);            
            hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);

            H5Pset_dxpl_mpio(xfer_plist_id, H5FD_MPIO_COLLECTIVE);
            H5Pset_all_coll_metadata_ops(xfer_plist_id, true);

            hid_t dspace_id = H5Dget_space(dset_id);

            // TODO check if this is needed
            if (!grid->localEmpty())
                H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, grid->getOffsets().data(), nullptr, localSize.data(), nullptr);
            else
                H5Sselect_none(dspace_id);

            hid_t mspace_id = H5Screate_simple(ndims, localSize.data(), nullptr);

            if (!grid->globalEmpty())
                H5Dread(dset_id, datatypeToHDF5type(channel.datatype), mspace_id, dspace_id, xfer_plist_id, channel.data);

            H5Sclose(mspace_id);
            H5Sclose(dspace_id);
            H5Pclose(xfer_plist_id);
            H5Dclose(dset_id);
        }

        void readData(hid_t file_id, const Grid* grid, std::vector<Channel>& channels)
        {
            for (auto& channel : channels) 
                readDataSet(file_id, grid, channel);
        }        
        
        void close(hid_t file_id)
        {
            H5Fclose(file_id);
        }
        
        void write(std::string filename, MPI_Comm comm, const Grid *grid, const std::vector<Channel>& channels)
        {
            auto file_id = create(filename, comm);
            if (file_id < 0)
            {
                if (file_id < 0) error("HDF5 failed to write to file '%s'", filename.c_str());
                return;
            }
            
            grid->write_to_HDF5(file_id, comm);
            writeData(file_id, grid, channels);
            
            close(file_id);
        }

        void read(std::string filename, MPI_Comm comm, Grid *grid, std::vector<Channel>& channels)
        {
            auto file_id = openReadOnly(filename, comm);
            if (file_id < 0)
            {
                if (file_id < 0) error("HDF5 failed to read from file '%s'", filename.c_str());
                return;
            }

            setbuf(stdout, NULL);
            
            grid->read_from_HDF5(file_id, comm);

            readData(file_id, grid, channels);
            
            close(file_id);
        }
    }
}
