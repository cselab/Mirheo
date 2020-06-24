// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "from_file.h"

#include <cassert>
#include <fstream>
#include <texture_types.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo
{

namespace interpolate_kernels
{
__device__ inline float cubicInterpolate1D(float y[4], float mu)
{
    // mu == 0 at y[1], mu == 1 at y[2]
    const float a0 = -0.5f*y[0] + 1.5f*y[1] - 1.5f*y[2] + 0.5f*y[3];
    const float a1 = y[0] - 2.5f*y[1] + 2.0f*y[2] - 0.5f*y[3];
    const float a2 = -0.5f*y[0] + 0.5f*y[2];
    const float a3 = y[1];

    return ((a0*mu + a1)*mu + a2)*mu + a3;
}

__global__ void cubicInterpolate3D(const float* in, int3 inDims, float3 inH, float* out, int3 outDims, float3 outH, float3 offset, float scalingFactor)
{
    // Inspired by http://paulbourke.net/miscellaneous/interpolation/
    // Origin of the output domain is in offset
    // Origin of the input domain is in (0,0,0)

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= outDims.x || iy >= outDims.y || iz >= outDims.z) return;

    float interp2D[4][4];
    float interp1D[4];

    // Coordinates where to interpolate
    float3 outputId  = make_float3(ix, iy, iz);
    float3 outputCoo = outputId*outH;

    float3 inputCoo  = outputCoo + offset;

    // Make sure we're within the region where the input data is defined
    assert( 0.0f <= inputCoo.x && inputCoo.x <= inDims.x*inH.x &&
            0.0f <= inputCoo.y && inputCoo.y <= inDims.y*inH.y &&
            0.0f <= inputCoo.z && inputCoo.z <= inDims.z*inH.z    );

    // Reference point of the original grid, rounded down
    int3 inputId_down = make_int3( math::floor(inputCoo / inH) );
    float3 mu = (inputCoo - make_float3(inputId_down)*inH) / inH;

    // Interpolate along x
    for (int dz = -1; dz <= 2; dz++)
        for (int dy = -1; dy <= 2; dy++)
        {
            float vals[4];

            for (int dx = -1; dx <= 2; dx++)
            {
                int3 delta{dx, dy, dz};
                const int3 curInputId = (inputId_down+delta + inDims) % inDims;

                vals[dx+1] = in[ (curInputId.z*inDims.y + curInputId.y) * inDims.x + curInputId.x ] * scalingFactor;
            }

            interp2D[dz+1][dy+1] = cubicInterpolate1D(vals, mu.x);
        }

    // Interpolate along y
    for (int dz = 0; dz <= 3; dz++)
        interp1D[dz] = cubicInterpolate1D(interp2D[dz], mu.y);

    // Interpolate along z
    out[ (iz*outDims.y + iy) * outDims.x + ix ] = cubicInterpolate1D(interp1D, mu.z);
}



__device__ inline float interpolationKernel(float3 x, float3 x0)
{
    //const int p = 8;
    const float3 r = x-x0;
    const float l2 = dot(r, r);
    const float l4 = l2*l2;

    return l4*l4;
}

__global__ void inverseDistanceWeightedInterpolation(const float* in, int3 inDims, float3 inH, float* out, int3 outDims, float3 outH, float3 offset, float scalingFactor)
{
    // Inspired by http://paulbourke.net/miscellaneous/interpolation/
    // Origin of the output domain is in offset
    // Origin of the input domain is in (0,0,0)

    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= outDims.x || iy >= outDims.y || iz >= outDims.z) return;

    // Coordinates where to interpolate
    float3 outputId  = make_float3(ix, iy, iz);
    float3 outputCoo = outputId*outH;

    float3 inputCoo  = outputCoo + offset;

    // Make sure we're within the region where the input data is defined
    assert( 0.0f <= inputCoo.x && inputCoo.x <= inDims.x*inH.x &&
            0.0f <= inputCoo.y && inputCoo.y <= inDims.y*inH.y &&
            0.0f <= inputCoo.z && inputCoo.z <= inDims.z*inH.z    );

    // Reference point of the original grid, rounded down
    int3 inputId_down = make_int3( math::floor(inputCoo / inH) );

    float nominator = 0, denominator = 0;

    // Interpolate along x
    for (int dz = -1; dz <= 2; dz++)
        for (int dy = -1; dy <= 2; dy++)
            for (int dx = -1; dx <= 2; dx++)
            {
                int3 delta{dx, dy, dz};
                const int3 curInputId = (inputId_down+delta + inDims) % inDims;
                const float3 curInputCoo = make_float3(curInputId)*inH;

                const float k = interpolationKernel(inputCoo, curInputCoo);
                nominator += in[ (curInputId.z*inDims.y + curInputId.y) * inDims.x + curInputId.x ] * k;
                denominator += k;
            }

    out[ (iz*outDims.y + iy) * outDims.x + ix ] = scalingFactor * nominator / denominator;
}
} // interpolate_kernels


struct HeaderInfo
{
    int3 resolution;
    float3 extents;
    int64_t fullSdfSize_byte;
    int64_t endHeader_byte;
};

inline auto multiplyComps(int3 v) {return v.x * v.y * v.z;}

inline int getRank(const MPI_Comm& comm)
{
    int rank;
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    return rank;
}

inline int getNranks(const MPI_Comm& comm)
{
    int size;
    MPI_Check( MPI_Comm_size(comm, &size) );
    return size;
}

static HeaderInfo readHeader(const std::string& fileName, const MPI_Comm& comm)
{
    HeaderInfo info;
    constexpr int root = 0;

    if (getRank(comm) == root)
    {
        std::ifstream file(fileName);
        if (!file.good())
            die("'%s': file not found or not accessible", fileName.c_str());

        auto fstart = file.tellg();

        file >> info.extents.x >> info.extents.y >> info.extents.z >>
            info.resolution.x >> info.resolution.y >> info.resolution.z;
        info.fullSdfSize_byte = (int64_t) multiplyComps(info.resolution) * sizeof(float);

        info("Using field file '%s' of size %.2fx%.2fx%.2f and resolution %dx%dx%d",
             fileName.c_str(), info.extents.x, info.extents.y, info.extents.z,
             info.resolution.x, info.resolution.y, info.resolution.z);

        file.seekg( 0, std::ios::end );
        auto fend = file.tellg();

        info.endHeader_byte = (fend - fstart) - info.fullSdfSize_byte;

        file.close();
    }

    MPI_Check( MPI_Bcast(&info.extents,          3, getMPIFloatType<float>(), root, comm) );
    MPI_Check( MPI_Bcast(&info.resolution,       3, MPI_INT,       root, comm) );
    MPI_Check( MPI_Bcast(&info.fullSdfSize_byte, 1, MPI_INT64_T,   root, comm) );
    MPI_Check( MPI_Bcast(&info.endHeader_byte,   1, MPI_INT64_T,   root, comm) );

    return info;
}

static std::vector<float> readSdf(const std::string& fileName, const MPI_Comm& comm, const HeaderInfo& info)
{
    const int rank   = getRank  (comm);
    const int nranks = getNranks(comm);

    // Read part and allgather
    const int64_t readPerProc_byte = (info.fullSdfSize_byte + nranks - 1) / (int64_t)nranks;
    std::vector<char> readBuffer(readPerProc_byte);

    // Limits in bytes
    const int64_t readStart = readPerProc_byte * rank + info.endHeader_byte;
    const int64_t readEnd   = std::min( readStart + readPerProc_byte, info.fullSdfSize_byte + info.endHeader_byte);

    MPI_File fh;
    MPI_Status status;
    MPI_Check( MPI_File_open(comm, fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) );  // TODO: MPI_Info
    MPI_Check( MPI_File_read_at_all(fh, readStart, readBuffer.data(), static_cast<int>(readEnd - readStart), MPI_BYTE, &status) );
    // TODO: check that we read just what we asked
    // MPI_Get_count only return int though

    const size_t n = readPerProc_byte * nranks / sizeof(float); // May be bigger than fullSdfSize, to make gather easier
    std::vector<float> fullSdfData(n);
    MPI_Check( MPI_Allgather(readBuffer.data(), static_cast<int>(readPerProc_byte), MPI_BYTE,
                             fullSdfData.data(), static_cast<int>(readPerProc_byte), MPI_BYTE, comm) );

    MPI_Check( MPI_File_close(&fh) );

    return fullSdfData;
}

struct LocalSdfPiece
{
    PinnedBuffer<float> data;
    float3 offset;
    int3 resolution;
};

static LocalSdfPiece prepareRelevantSdfPiece(const std::vector<float>& fullSdfData, float3 extendedDomainStart, float3 extendedDomainSize,
                                             float3 initialSdfH, int3 initialSdfResolution)
{
    LocalSdfPiece sdfPiece;
    // Find your relevant chunk of data
    // We cannot send big sdf files directly, so we'll carve a piece now

    constexpr int margin = 3; // +2 from cubic interpolation, +1 from possible round-off errors
    const int3 startId = make_int3( math::floor( extendedDomainStart                     / initialSdfH) ) - margin;
    const int3 endId   = make_int3( math::ceil ((extendedDomainStart+extendedDomainSize) / initialSdfH) ) + margin;

    const float3 startInLocalCoord = make_float3(startId)*initialSdfH - (extendedDomainStart + 0.5*extendedDomainSize);

    sdfPiece.offset = -0.5*extendedDomainSize - startInLocalCoord;
    sdfPiece.resolution = endId - startId;

    sdfPiece.data.resize_anew( multiplyComps(sdfPiece.resolution) );

    auto locSdfDataPtr = sdfPiece.data.hostPtr();

    for (int k = 0; k < sdfPiece.resolution.z; ++k)
        for (int j = 0; j < sdfPiece.resolution.y; ++j)
            for (int i = 0; i < sdfPiece.resolution.x; ++i)
            {
                const int origIx = (i+startId.x + initialSdfResolution.x) % initialSdfResolution.x;
                const int origIy = (j+startId.y + initialSdfResolution.y) % initialSdfResolution.y;
                const int origIz = (k+startId.z + initialSdfResolution.z) % initialSdfResolution.z;

                const auto dstId = (k*sdfPiece.resolution.y + j)*sdfPiece.resolution.x + i;
                const auto srcId = (origIz*initialSdfResolution.y + origIy)*initialSdfResolution.x + origIx;
                locSdfDataPtr[ dstId ] = fullSdfData[ srcId ];
            }
    return sdfPiece;
}

FieldFromFile::FieldFromFile(const MirState *state, std::string name, std::string fieldFileName, real3 h) :
    Field(state, name, h),
    fieldFileName_(fieldFileName)
{}

FieldFromFile::~FieldFromFile() = default;

FieldFromFile::FieldFromFile(FieldFromFile&&) = default;

inline bool componentsAreEqual(float3 v, float eps = 1e-5f)
{
    return math::abs(v.x - v.y) < eps && math::abs(v.x - v.z) < eps;
}

void FieldFromFile::setup(const MPI_Comm& comm)
{
    info("Setting up field from %s", fieldFileName_.c_str());

    const auto domain = getState()->domain;

    CUDA_Check( cudaDeviceSynchronize() );

    int nranks, rank;
    MPI_Check( MPI_Comm_size(comm, &nranks) );
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    // Read header
    auto headerInfo = readHeader(fieldFileName_, comm);
    const float3 initialSdfH = make_float3(domain.globalSize) / make_float3(headerInfo.resolution-1);

    // Read heavy data
    const auto fullSdfData = readSdf(fieldFileName_, comm, headerInfo);

    const float3 scale3 = make_float3(domain.globalSize) / headerInfo.extents;
    if ( !componentsAreEqual(scale3) )
        die("Sdf size and domain size mismatch");
    const float lenScalingFactor = (scale3.x + scale3.y + scale3.z) / 3;

    auto sdfPiece = prepareRelevantSdfPiece(fullSdfData,
                                            make_float3(domain.globalStart - margin3_), make_float3(extendedDomainSize_),
                                            initialSdfH, headerInfo.resolution);

    // Interpolate
    DeviceBuffer<float> fieldRawData (multiplyComps(resolution_));

    const dim3 threads(8, 8, 8);
    const dim3 blocks((resolution_.x+threads.x-1) / threads.x,
                      (resolution_.y+threads.y-1) / threads.y,
                      (resolution_.z+threads.z-1) / threads.z);

    sdfPiece.data.uploadToDevice(defaultStream);
    SAFE_KERNEL_LAUNCH(
            interpolate_kernels::cubicInterpolate3D,
            blocks, threads, 0, defaultStream,
            sdfPiece.data.devPtr(), sdfPiece.resolution, initialSdfH,
            fieldRawData.devPtr(), resolution_, make_float3(h_),
            sdfPiece.offset, lenScalingFactor );

    _setupArrayTexture(fieldRawData.devPtr());
}

} // namespace mirheo
