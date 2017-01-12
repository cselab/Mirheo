#include <hdf5.h>
#include <regex>

#include "write_xdmf.h"

void XDMFDumper::writeLight(std::string fname, float t)
{
	FILE* xmf;
	fout = fopen( (fname+".xmf").c_str() );
	if (fout == nullptr)
	{
		if (myrank == 0) error("XMF write failed: %s", (fname+".xmf").c_str());
		return;
	}

	fprintf(xmf, "<?xml version=\"1.0\" ?>\n");
	fprintf(xmf, "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n");
	fprintf(xmf, "<Xdmf Version=\"2.0\">\n");
	fprintf(xmf, " <Domain>\n");
	fprintf(xmf, "   <Grid Name=\"mesh\" GridType=\"Uniform\">\n");
	fprintf(xmf, "     <Time Value=\"%.f\"/>\n", t);
	fprintf(xmf, "     <Topology TopologyType=\"3DCORECTMesh\" Dimensions=\"%d %d %d\"/>\n",
			dimensions.x+1, dimensions.y+1, dimensions.z+1);

	fprintf(xmf, "     <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n");
	fprintf(xmf, "       <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");
	fprintf(xmf, "        %e %e %e\n", 0.0, 0.0, 0.0);
	fprintf(xmf, "       </DataItem>\n");
	fprintf(xmf, "       <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Float\" Precision=\"4\" Format=\"XML\">\n");

	fprintf(xmf, "        %e %e %e\n", h.x, h.y, h.z);
	fprintf(xmf, "       </DataItem>\n");
	fprintf(xmf, "     </Geometry>\n");

	for(int ichannel = 0; ichannel < channelNames.size(); ichannel++)
	{
		fprintf(xmf, "     <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Cell\">\n", channelNames[ichannel].c_str());
		fprintf(xmf, "       <DataItem Dimensions=\"%d %d %d 1\" NumberType=\"Float\" Precision=\"4\" Format=\"HDF\">\n",
				dimensions.x, dimensions.y, dimensions.z);

		fprintf(xmf, "        %s:/%s\n", (fname+".h5").c_str(), channelNames[ichannel].c_str());

		fprintf(xmf, "       </DataItem>\n");
		fprintf(xmf, "     </Attribute>\n");
	}

	fprintf(xmf, "   </Grid>\n");
	fprintf(xmf, " </Domain>\n");
	fprintf(xmf, "</Xdmf>\n");

	fclose(xmf);
}

void XDMFDumper::writeHeavy(std::string fname, std::vector<float*> channelData)
{
	id_t plist_id_access = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id_access, xdmfcomm, MPI_INFO_NULL);

	hid_t file_id = H5Fcreate( (fname+".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id_access );
	if (file_id < 0)
	{
		if (myrank == 0) error("HDF5 write failed: %s", (fname+".h5").c_str());
		return;
	}

	H5Pclose(plist_id_access);

	hsize_t globalsize[4] = {nranks[2] * dimensions.z, nranks[1] * dimensions.y, nranks[0] * dimensions.x, 1};
	hid_t filespace_simple = H5Screate_simple(4, globalsize, nullptr);

	for(int ichannel = 0; ichannel < channelNames.size(); ++ichannel)
	{
		hid_t dset_id = H5Dcreate(file_id, channelNames[ichannel].c_str(), H5T_NATIVE_FLOAT, filespace_simple, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		id_t plist_id = H5Pcreate(H5P_DATASET_XFER);

		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

		hsize_t start[4] = { myrank[2] * dimensions.z, myrank[1] * dimensions.y, myrank[0] * dimensions.x, 0 };
		hsize_t extent[4] = { dimensions.z, dimensions.y, dimensions.x, 1 };
		hid_t filespace = H5Dget_space(dset_id);
		H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, extent, NULL);

		hid_t memspace = H5Screate_simple(4, extent, NULL);
		herr_t status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, channelData[ichannel]);

		H5Sclose(memspace);
		H5Sclose(filespace);
		H5Pclose(plist_id);
		H5Dclose(dset_id);
	}

	H5Sclose(filespace_simple);
	H5Fclose(file_id);
}

XDMFDumper::XDMFDumper(MPI_Comm comm, std::string path, int3 dimensions, float3 h, std::vector<std::string> channelNames) :
		path(path), dimensions(dimensions), h(h), channelNames(channelNames), deactivated(false), timeStamp(0)
{
	MPI_Check( MPI_Comm_dup(comm, &haloComm));
	MPI_Check( MPI_Cart_get(xdmfcomm, 3, nranks, periods, my3Drank) );
	MPI_Check( MPI_Comm_rank(xdmfcomm, &myrank));

	// Create folder
	if (myrank == 0)
	{
		std::regex re("^(.*)\/.+");
		std::smatch match;
		if (std::regex_match(path, match, re))
		{
			std::string folders = match[1].str();
			std::string command = "mkdir -p " + folders;
			if ( system(command.c_str()) != 0 )
			{
				error("Could not create folders or files by given path, dumping will be disabled.");
				deactivated = true;
			}
		}
	}
}

void XDMFDumper::dump(std::vector<float*> channelData, const float t)
{
	if (deactivated) return;

	std::string tstr = to_string(timeStamp++);
	std::string fname = std::string(padding - tstr.length(), '0') + tstr;

	if (myrank == 0) writeLight(fname, t);
	writeHeavy(fname, channelData);

	if (myrank == 0) debug("XDMF write successful: %s", (fname+"[.h5 .xmf]").c_str());
}
