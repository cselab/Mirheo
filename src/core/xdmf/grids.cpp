#include "grids.h"
#include "common.h"
#include "hdf5_helpers.h"
#include "type_map.h"

#include <core/logger.h>

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
int  GridDims::getDims()     const { return getLocalSize().size();         }
    
//
// Uniform Grid
//

UniformGrid::UniformGridDims::UniformGridDims(int3 localSize, MPI_Comm cartComm)
{
    int nranks[3], periods[3], my3Drank[3];
    MPI_Check( MPI_Cart_get(cartComm, 3, nranks, periods, my3Drank) );

    this->localSize  = std::vector<hsize_t>{ (hsize_t)localSize.x,  (hsize_t)localSize.y,  (hsize_t)localSize.z};
        
    this->globalSize = std::vector<hsize_t>{ (hsize_t) nranks[0] * localSize.x,
                                             (hsize_t) nranks[1] * localSize.y,
                                             (hsize_t) nranks[2] * localSize.z};

    this->offsets   = std::vector<hsize_t>{ (hsize_t) my3Drank[2] * localSize.z,
                                            (hsize_t) my3Drank[1] * localSize.y,
                                            (hsize_t) my3Drank[0] * localSize.x,
                                            (hsize_t) 0 };
}

std::vector<hsize_t> UniformGrid::UniformGridDims::getLocalSize()  const {return localSize;}
std::vector<hsize_t> UniformGrid::UniformGridDims::getGlobalSize() const {return globalSize;}
std::vector<hsize_t> UniformGrid::UniformGridDims::getOffsets()    const {return offsets;}
    
std::string UniformGrid::getCentering() const                        { return "Cell"; }
const UniformGrid::UniformGridDims* UniformGrid::getGridDims() const { return &dims; }
    
void UniformGrid::writeToHDF5(__UNUSED hid_t file_id, __UNUSED MPI_Comm comm) const
{}
    
pugi::xml_node UniformGrid::writeToXMF(pugi::xml_node node, __UNUSED std::string h5filename) const
{
    auto gridNode = node.append_child("Grid");
    gridNode.append_attribute("Name") = "mesh";
    gridNode.append_attribute("GridType") = "Uniform";
        
    // Topology size is in vertices, so it's +1 wrt to the number of cells
    auto nodeResolution = dims.globalSize;
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
    spaceNode.text() = to_string(spacing).c_str();
        
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
        
UniformGrid::UniformGrid(int3 localSize, real3 h, MPI_Comm cartComm) :
    dims(localSize, cartComm),
    spacing{h.x, h.y, h.z}
{}
    
//
// Vertex Grid
//

VertexGrid::VertexGridDims::VertexGridDims(long nlocal, MPI_Comm comm) :
    nlocal(nlocal),
    nglobal(0),
    offset(0)
{
    MPI_Check( MPI_Exscan   (&nlocal, &offset,  1, MPI_LONG_LONG_INT, MPI_SUM, comm) );
    MPI_Check( MPI_Allreduce(&nlocal, &nglobal, 1, MPI_LONG_LONG_INT, MPI_SUM, comm) );    
}

VertexGrid::VertexGrid(std::shared_ptr<std::vector<real3>> positions, MPI_Comm comm) :
    dims(positions->size(), comm),
    positions(positions)
{}

std::vector<hsize_t> VertexGrid::VertexGridDims::getLocalSize()  const {return {nlocal};}
std::vector<hsize_t> VertexGrid::VertexGridDims::getGlobalSize() const {return {nglobal};}
std::vector<hsize_t> VertexGrid::VertexGridDims::getOffsets()    const {return {offset, 0};}

const VertexGrid::VertexGridDims* VertexGrid::getGridDims() const    { return &dims; }    
std::string VertexGrid::getCentering() const                         { return "Node"; }    

void VertexGrid::writeToHDF5(hid_t file_id, __UNUSED MPI_Comm comm) const
{
    Channel posCh(positionChannelName, (void*) positions->data(),
                  Channel::DataForm::Vector, XDMF::getNumberType<real>(),
                  DataTypeWrapper<real>(), Channel::NeedShift::True);
        
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
    partNode.append_attribute("Dimensions") = (std::to_string(dims.nglobal) + " 3").c_str();
    partNode.append_attribute("NumberType") = numberTypeToString(XDMF::getNumberType<real>()).c_str();
    partNode.append_attribute("Precision") = std::to_string(numberTypeToPrecision(XDMF::getNumberType<real>())).c_str();
    partNode.append_attribute("Format") = "HDF";
    partNode.text() = (h5filename + ":/" + positionChannelName).c_str();
        
    return gridNode;
}

void VertexGrid::readFromXMF(const pugi::xml_node &node, std::string &h5filename)
{
    int d = 0;

    auto partNode = node.child("Geometry").child("DataItem");

    if (!partNode)
        die("Wrong format");
        
    std::istringstream dimensions( partNode.attribute("Dimensions").value() );

    dimensions >> dims.nglobal;
    dimensions >> d;

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

    int64_t nchunksGlobal = dims.nglobal / chunkSize;
    if (nchunksGlobal * chunkSize != static_cast<int64_t>(dims.nglobal))
        die("incompatible chunk size");

    int64_t nchunksLocal = (nchunksGlobal + size - 1) / size;
    int64_t chunksOffset = nchunksLocal * rank;

    // Don't read past the file size
    chunksOffset = std::min(chunksOffset, nchunksGlobal);

    if (chunksOffset + nchunksLocal > nchunksGlobal)
        nchunksLocal = std::max(nchunksGlobal - chunksOffset, 0l);

    dims.nlocal = nchunksLocal * chunkSize;
    dims.offset = chunksOffset * chunkSize;
}

void VertexGrid::readFromHDF5(hid_t file_id, __UNUSED MPI_Comm comm)
{
    positions->resize(dims.nlocal);
    Channel posCh(positionChannelName, positions->data(), Channel::DataForm::Vector,
                  XDMF::getNumberType<real>(), DataTypeWrapper<real>(), Channel::NeedShift::True);
        
    HDF5::readDataSet(file_id, getGridDims(), posCh);
}

void VertexGrid::_writeTopology(pugi::xml_node& topoNode, __UNUSED std::string h5filename) const
{
    topoNode.append_attribute("TopologyType") = "Polyvertex";
    topoNode.append_attribute("NumberOfElements") = std::to_string(dims.nglobal).c_str();    
}

const std::string VertexGrid::positionChannelName = "position";

//
// Triangle Mesh Grid
//
    
TriangleMeshGrid::TriangleMeshGrid(std::shared_ptr<std::vector<real3>> positions,
                                   std::shared_ptr<std::vector<int3>> triangles,
                                   MPI_Comm comm) :
    VertexGrid(positions, comm),
    dimsTriangles(triangles->size(), comm),
    triangles(triangles)
{}

void TriangleMeshGrid::writeToHDF5(hid_t file_id, MPI_Comm comm) const
{
    VertexGrid::writeToHDF5(file_id, comm);

    Channel triCh(triangleChannelName, (void*) triangles->data(),
                  Channel::DataForm::Triangle, Channel::NumberType::Int,
                  DataTypeWrapper<int>(), Channel::NeedShift::False);

    HDF5::writeDataSet(file_id, &dimsTriangles, triCh);
}
                
void TriangleMeshGrid::_writeTopology(pugi::xml_node& topoNode, std::string h5filename) const
{
    topoNode.append_attribute("TopologyType") = "Triangle";

    topoNode.append_attribute("NumberOfElements") = std::to_string(dimsTriangles.nglobal).c_str();

    auto triangleNode = topoNode.append_child("DataItem");

    triangleNode.append_attribute("Dimensions") = (std::to_string(dimsTriangles.nglobal) + " 3").c_str();
    triangleNode.append_attribute("NumberType") = numberTypeToString(Channel::NumberType::Int).c_str();
    triangleNode.append_attribute("Precision") = std::to_string(numberTypeToPrecision(Channel::NumberType::Int)).c_str();
    triangleNode.append_attribute("Format") = "HDF";
    triangleNode.text() = (h5filename + ":/" + triangleChannelName).c_str();

}

const std::string TriangleMeshGrid::triangleChannelName = "triangle";

} // namespace XDMF
