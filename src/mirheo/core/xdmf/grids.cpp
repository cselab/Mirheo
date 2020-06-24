// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "grids.h"
#include "common.h"
#include "hdf5_helpers.h"
#include "type_map.h"

#include <mirheo/core/logger.h>

namespace mirheo
{

namespace XDMF
{
static hsize_t product(const std::vector<hsize_t>& v)
{
    hsize_t prod = 1;
    for (auto d : v)
        prod *= d;
    return prod;
}

bool GridDims::localEmpty()  const { return product(getLocalSize ()) == 0; }
bool GridDims::globalEmpty() const { return product(getGlobalSize()) == 0; }
int  GridDims::getDims()     const { return (int) getLocalSize().size();   }

//
// Uniform Grid
//

UniformGrid::UniformGrid(int3 localSize, real3 h, MPI_Comm cartComm) :
    dims_(localSize, cartComm),
    spacing_{h.x, h.y, h.z}
{}

UniformGrid::UniformGridDims::UniformGridDims(int3 localSize, MPI_Comm cartComm)
{
    int nranks[3], periods[3], my3Drank[3];
    MPI_Check( MPI_Cart_get(cartComm, 3, nranks, periods, my3Drank) );

    localSize_  = std::vector<hsize_t>{ (hsize_t)localSize.x,  (hsize_t)localSize.y,  (hsize_t)localSize.z};

    globalSize_ = std::vector<hsize_t>{ (hsize_t) nranks[0] * localSize.x,
                                        (hsize_t) nranks[1] * localSize.y,
                                        (hsize_t) nranks[2] * localSize.z};

    offsets_   = std::vector<hsize_t>{ (hsize_t) my3Drank[2] * localSize.z,
                                       (hsize_t) my3Drank[1] * localSize.y,
                                       (hsize_t) my3Drank[0] * localSize.x,
                                       (hsize_t) 0 };
}

std::vector<hsize_t> UniformGrid::UniformGridDims::getLocalSize()  const {return localSize_;}
std::vector<hsize_t> UniformGrid::UniformGridDims::getGlobalSize() const {return globalSize_;}
std::vector<hsize_t> UniformGrid::UniformGridDims::getOffsets()    const {return offsets_;}

std::string UniformGrid::getCentering() const { return "Cell"; }
const GridDims* UniformGrid::getGridDims() const { return &dims_; }

void UniformGrid::writeToHDF5(__UNUSED hid_t file_id, __UNUSED MPI_Comm comm) const
{}

pugi::xml_node UniformGrid::writeToXMF(pugi::xml_node node, __UNUSED std::string h5filename) const
{
    auto gridNode = node.append_child("Grid");
    gridNode.append_attribute("Name") = "mesh";
    gridNode.append_attribute("GridType") = "Uniform";

    // Topology size is in vertices, so it's +1 wrt to the number of cells
    auto nodeResolution = dims_.getGlobalSize();
    for (auto& r : nodeResolution) r += 1;

    // One more What. The. F
    std::reverse(nodeResolution.begin(), nodeResolution.end());

    auto topoNode = gridNode.append_child("Topology");
    topoNode.append_attribute("TopologyType") = "3DCORECTMesh";
    topoNode.append_attribute("Dimensions") = to_string(nodeResolution).c_str();

    auto geomNode = gridNode.append_child("Geometry");
    geomNode.append_attribute("GeometryType") = "ORIGIN_DXDYDZ";


    auto setup3Fnode = [] (pugi::xml_node& n, std::string nodeName)
    {
        const std::array< std::pair<std::string, std::string>, 5 > name_vals =
            {{
              { "Name", nodeName },
              { "Dimensions", "3" },
              { "NumberType", numberTypeToString(Channel::NumberType::Float) },
              { "Precision", std::to_string(numberTypeToPrecision(Channel::NumberType::Float)) },
              { "Format", "XML" }
              }};

        for (auto& name_val : name_vals)
            n.append_attribute(name_val.first.c_str()) = name_val.second.c_str();
    };

    auto origNode = geomNode.append_child("DataItem");
    setup3Fnode(origNode, "Origin");
    origNode.text() = "0.0 0.0 0.0";

    auto spaceNode = geomNode.append_child("DataItem");
    setup3Fnode(spaceNode, "Spacing");
    spaceNode.text() = to_string(spacing_).c_str();

    return gridNode;
}

void UniformGrid::readFromXMF(__UNUSED const pugi::xml_node &node, __UNUSED std::string &h5filename)
{
    // TODO
    die("not implemented");
}

void UniformGrid::splitReadAccess(__UNUSED MPI_Comm comm, __UNUSED int chunkSize)
{
    // TODO
    die("not implemented");
}

void UniformGrid::readFromHDF5(__UNUSED hid_t file_id, __UNUSED MPI_Comm comm)
{
    // TODO
    die("not implemented");
}

//
// Vertex Grid
//

VertexGrid::VertexGridDims::VertexGridDims(long nLocal, MPI_Comm comm) :
    nLocal_(nLocal),
    nGlobal_(0),
    offset_(0)
{
    MPI_Check( MPI_Exscan   (&nLocal_, &offset_,  1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(&nLocal_, &nGlobal_, 1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
}

std::vector<hsize_t> VertexGrid::VertexGridDims::getLocalSize()  const {return {nLocal_};}
std::vector<hsize_t> VertexGrid::VertexGridDims::getGlobalSize() const {return {nGlobal_};}
std::vector<hsize_t> VertexGrid::VertexGridDims::getOffsets()    const {return {offset_, 0};}

hsize_t VertexGrid::VertexGridDims::getNLocal()  const {return nLocal_;}
void VertexGrid::VertexGridDims::setNLocal(hsize_t n) {nLocal_ = n;}

hsize_t VertexGrid::VertexGridDims::getNGlobal()  const {return nGlobal_;}
void VertexGrid::VertexGridDims::setNGlobal(hsize_t n) {nGlobal_ = n;}

void VertexGrid::VertexGridDims::setOffset(hsize_t n) {offset_ = n;}



VertexGrid::VertexGrid(std::shared_ptr<std::vector<real3>> positions, MPI_Comm comm) :
    dims_(positions->size(), comm),
    positions_(positions)
{}

const GridDims* VertexGrid::getGridDims() const { return &dims_; }
std::string VertexGrid::getCentering() const    { return "Node"; }

void VertexGrid::writeToHDF5(hid_t file_id, __UNUSED MPI_Comm comm) const
{
    Channel posCh {positionChannelName_, (void*) positions_->data(),
                   Channel::DataForm::Vector, XDMF::getNumberType<real>(),
                   DataTypeWrapper<real>(), Channel::NeedShift::True};

    HDF5::writeDataSet(file_id, getGridDims(), posCh);
}

pugi::xml_node VertexGrid::writeToXMF(pugi::xml_node node, std::string h5filename) const
{
    auto gridNode = node.append_child("Grid");
    gridNode.append_attribute("Name") = "mesh";
    gridNode.append_attribute("GridType") = "Uniform";

    auto topoNode = gridNode.append_child("Topology");
    _writeTopology(topoNode, h5filename);

    auto geomNode = gridNode.append_child("Geometry");
    geomNode.append_attribute("GeometryType") = "XYZ";

    auto partNode = geomNode.append_child("DataItem");
    partNode.append_attribute("Dimensions") = (std::to_string(dims_.getNGlobal()) + " 3").c_str();
    partNode.append_attribute("NumberType") = numberTypeToString(XDMF::getNumberType<real>()).c_str();
    partNode.append_attribute("Precision") = std::to_string(numberTypeToPrecision(XDMF::getNumberType<real>())).c_str();
    partNode.append_attribute("Format") = "HDF";
    partNode.text() = (h5filename + ":/" + positionChannelName_).c_str();

    return gridNode;
}

void VertexGrid::readFromXMF(const pugi::xml_node &node, std::string &h5filename)
{
    int d {0};
    hsize_t nglobal {0};

    auto partNode = node.child("Geometry").child("DataItem");

    if (!partNode)
        die("Wrong format");

    std::istringstream dimensions( partNode.attribute("Dimensions").value() );

    dimensions >> nglobal;
    dimensions >> d;
    dims_.setNGlobal(nglobal);

    if (d != 3)
        die("expected 3 dimesnional positions, got %d.", d);

    std::string positionDataSet(partNode.text().as_string());
    auto endH5 = positionDataSet.find(":");

    if (endH5 == std::string::npos)
        die("expected dataset name from h5 file: got %s", positionDataSet.c_str());

    h5filename = positionDataSet.substr(0, endH5);
}

void VertexGrid::splitReadAccess(MPI_Comm comm, int chunkSize)
{
    int size, rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    MPI_Check( MPI_Comm_size(comm, &size) );

    const int64_t nchunksGlobal = dims_.getNGlobal() / chunkSize;
    if (nchunksGlobal * chunkSize != static_cast<int64_t>(dims_.getNGlobal()))
        die("incompatible chunk size");

    int64_t nchunksLocal = (nchunksGlobal + size - 1) / size;
    int64_t chunksOffset = nchunksLocal * rank;

    // Don't read past the file size
    chunksOffset = std::min(chunksOffset, nchunksGlobal);

    if (chunksOffset + nchunksLocal > nchunksGlobal)
        nchunksLocal = std::max(nchunksGlobal - chunksOffset, 0l);

    dims_.setNLocal(nchunksLocal * chunkSize);
    dims_.setOffset(chunksOffset * chunkSize);
}

void VertexGrid::readFromHDF5(hid_t file_id, __UNUSED MPI_Comm comm)
{
    positions_->resize(dims_.getNLocal());
    Channel posCh {positionChannelName_, positions_->data(), Channel::DataForm::Vector,
                   XDMF::getNumberType<real>(), DataTypeWrapper<real>(), Channel::NeedShift::True};

    HDF5::readDataSet(file_id, getGridDims(), posCh);
}

void VertexGrid::_writeTopology(pugi::xml_node& topoNode, __UNUSED const std::string& h5filename) const
{
    topoNode.append_attribute("TopologyType") = "Polyvertex";
    topoNode.append_attribute("NumberOfElements") = std::to_string(dims_.getNGlobal()).c_str();
}

const std::string VertexGrid::positionChannelName_ = "position";

//
// Triangle Mesh Grid
//

TriangleMeshGrid::TriangleMeshGrid(std::shared_ptr<std::vector<real3>> positions,
                                   std::shared_ptr<std::vector<int3>> triangles,
                                   MPI_Comm comm) :
    VertexGrid(positions, comm),
    dimsTriangles_(triangles->size(), comm),
    triangles_(triangles)
{}

void TriangleMeshGrid::writeToHDF5(hid_t file_id, MPI_Comm comm) const
{
    VertexGrid::writeToHDF5(file_id, comm);

    Channel triCh {triangleChannelName_, (void*) triangles_->data(),
                   Channel::DataForm::Triangle, Channel::NumberType::Int,
                   DataTypeWrapper<int>(), Channel::NeedShift::False};

    HDF5::writeDataSet(file_id, &dimsTriangles_, triCh);
}

void TriangleMeshGrid::_writeTopology(pugi::xml_node& topoNode, const std::string& h5filename) const
{
    topoNode.append_attribute("TopologyType") = "Triangle";

    topoNode.append_attribute("NumberOfElements") = std::to_string(dimsTriangles_.getNGlobal()).c_str();

    auto triangleNode = topoNode.append_child("DataItem");

    triangleNode.append_attribute("Dimensions") = (std::to_string(dimsTriangles_.getNGlobal()) + " 3").c_str();
    triangleNode.append_attribute("NumberType") = numberTypeToString(Channel::NumberType::Int).c_str();
    triangleNode.append_attribute("Precision") = std::to_string(numberTypeToPrecision(Channel::NumberType::Int)).c_str();
    triangleNode.append_attribute("Format") = "HDF";
    triangleNode.text() = (h5filename + ":/" + triangleChannelName_).c_str();

}

const std::string TriangleMeshGrid::triangleChannelName_ = "triangle";

} // namespace XDMF

} // namespace mirheo
