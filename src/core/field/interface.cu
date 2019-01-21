#include "interface.h"

#include <fstream>
#include <cmath>
#include <texture_types.h>
#include <cassert>

#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>
#include <core/pvs/particle_vector.h>

namespace InterpolateKernels
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
    int3 inputId_down = make_int3( floorf(inputCoo / inH) );
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
    int3 inputId_down = make_int3( floorf(inputCoo / inH) );
    
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
} // InterpolateKernels


/*
 * We only set a few params here
 */
Field::Field(std::string fieldFileName, float3 h) :
    fieldFileName(fieldFileName),
    fieldArray(nullptr)
{
    this->h = h;
}

Field::~Field()
{
    if (fieldArray) {
        CUDA_Check( cudaFreeArray(fieldArray) );
        CUDA_Check( cudaDestroyTextureObject(fieldTex) );
    }
}

Field::Field(Field&&) = default;

const FieldDeviceHandler& Field::handler() const
{
    return *(FieldDeviceHandler*)this;
}

static void readHeader(const std::string fileName, const MPI_Comm& comm, int3& sdfResolution, float3& sdfExtent, int64_t& fullSdfSize_byte, int64_t& endHeader_byte, int rank)
{
    if (rank == 0)
    {
        std::ifstream file(fileName);
        if (!file.good())
            die("File not found or not accessible");

        auto fstart = file.tellg();

        file >> sdfExtent.x >> sdfExtent.y >> sdfExtent.z >>
            sdfResolution.x >> sdfResolution.y >> sdfResolution.z;
        fullSdfSize_byte = (int64_t)sdfResolution.x * sdfResolution.y * sdfResolution.z * sizeof(float);

        info("Using field file '%s' of size %.2fx%.2fx%.2f and resolution %dx%dx%d", fileName.c_str(),
                sdfExtent.x, sdfExtent.y, sdfExtent.z,
                sdfResolution.x, sdfResolution.y, sdfResolution.z);

        file.seekg( 0, std::ios::end );
        auto fend = file.tellg();

        endHeader_byte = (fend - fstart) - fullSdfSize_byte;

        file.close();
    }

    MPI_Check( MPI_Bcast(&sdfExtent,        3, MPI_FLOAT,     0, comm) );
    MPI_Check( MPI_Bcast(&sdfResolution,    3, MPI_INT,       0, comm) );
    MPI_Check( MPI_Bcast(&fullSdfSize_byte, 1, MPI_INT64_T,   0, comm) );
    MPI_Check( MPI_Bcast(&endHeader_byte,   1, MPI_INT64_T,   0, comm) );
}

static void readSdf(const std::string fileName, const MPI_Comm& comm, int64_t fullSdfSize_byte, int64_t endHeader_byte, int nranks, int rank, std::vector<float>& fullSdfData)
{
    // Read part and allgather
    const int64_t readPerProc_byte = (fullSdfSize_byte + nranks - 1) / (int64_t)nranks;
    std::vector<char> readBuffer(readPerProc_byte);

    // Limits in bytes
    const int64_t readStart = readPerProc_byte * rank + endHeader_byte;
    const int64_t readEnd   = std::min( readStart + readPerProc_byte, fullSdfSize_byte + endHeader_byte);

    MPI_File fh;
    MPI_Status status;
    MPI_Check( MPI_File_open(comm, fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) );  // TODO: MPI_Info
    MPI_Check( MPI_File_read_at_all(fh, readStart, readBuffer.data(), readEnd - readStart, MPI_BYTE, &status) );
    // TODO: check that we read just what we asked
    // MPI_Get_count only return int though

    fullSdfData.resize(readPerProc_byte * nranks / sizeof(float));  // May be bigger than fullSdfSize, to make gather easier
    MPI_Check( MPI_Allgather(readBuffer.data(), readPerProc_byte, MPI_BYTE, fullSdfData.data(), readPerProc_byte, MPI_BYTE, comm) );
}

static void prepareRelevantSdfPiece(const float* fullSdfData, float3 extendedDomainStart, float3 extendedDomainSize,
                                    float3 initialSdfH, int3 initialSdfResolution,
                                    int3& resolution, float3& offset, PinnedBuffer<float>& localSdfData)
{
    // Find your relevant chunk of data
    // We cannot send big sdf files directly, so we'll carve a piece now

    const int margin = 3; // +2 from cubic interpolation, +1 from possible round-off errors
    const int3 startId = make_int3( floorf( extendedDomainStart                     / initialSdfH) ) - margin;
    const int3 endId   = make_int3( ceilf ((extendedDomainStart+extendedDomainSize) / initialSdfH) ) + margin;

    float3 startInLocalCoord = make_float3(startId)*initialSdfH - (extendedDomainStart + 0.5*extendedDomainSize);
    offset = -0.5*extendedDomainSize - startInLocalCoord;

    resolution = endId - startId;

    localSdfData.resize( resolution.x * resolution.y * resolution.z, 0 );
    auto locSdfDataPtr = localSdfData.hostPtr();

    for (int k = 0; k < resolution.z; k++)
        for (int j = 0; j < resolution.y; j++)
            for (int i = 0; i < resolution.x; i++)
            {
                const int origIx = (i+startId.x + initialSdfResolution.x) % initialSdfResolution.x;
                const int origIy = (j+startId.y + initialSdfResolution.y) % initialSdfResolution.y;
                const int origIz = (k+startId.z + initialSdfResolution.z) % initialSdfResolution.z;

                locSdfDataPtr[ (k*resolution.y + j)*resolution.x + i ] =
                        fullSdfData[ (origIz*initialSdfResolution.y + origIy)*initialSdfResolution.x + origIx ];
            }
}

void Field::setup(MPI_Comm& comm, DomainInfo domain)
{
    info("Setting up field from %s", fieldFileName.c_str());

    CUDA_Check( cudaDeviceSynchronize() );
    MPI_Check( MPI_Comm_dup(comm, &comm) );

    int nranks, rank;
    int ranks[3], periods[3], coords[3];
    MPI_Check( MPI_Comm_size(comm, &nranks) );
    MPI_Check( MPI_Comm_rank(comm, &rank) );
    MPI_Check( MPI_Cart_get (comm, 3, ranks, periods, coords) );

    int3 initialSdfResolution;
    float3 initialSdfExtent;

    int64_t fullSdfSize_byte;
    int64_t endHeader_byte;

    // Read header
    readHeader(fieldFileName, comm, initialSdfResolution, initialSdfExtent, fullSdfSize_byte, endHeader_byte, rank);
    float3 initialSdfH = domain.globalSize / make_float3(initialSdfResolution-1);

    // Read heavy data
    std::vector<float> fullSdfData;
    readSdf(fieldFileName, comm, fullSdfSize_byte, endHeader_byte, nranks, rank, fullSdfData);

    // We'll make sdf a bit bigger, so that particles that flew away
    // would also be correctly bounced back
    extendedDomainSize = domain.localSize + 2.0f*margin3;
    resolution         = make_int3( ceilf(extendedDomainSize / h) );
    h                  = extendedDomainSize / make_float3(resolution-1);
    invh               = 1.0f / h;

    const float3 scale3 = domain.globalSize / initialSdfExtent;
    if ( fabs(scale3.x - scale3.y) > 1e-5 || fabs(scale3.x - scale3.z) > 1e-5 )
        die("Sdf size and domain size mismatch");
    const float lenScalingFactor = (scale3.x + scale3.y + scale3.z) / 3;

    int3 resolutionBeforeInterpolation;
    float3 offset;
    PinnedBuffer<float> localData;
    prepareRelevantSdfPiece(fullSdfData.data(), domain.globalStart - margin3, extendedDomainSize,
                            initialSdfH, initialSdfResolution,
                            resolutionBeforeInterpolation, offset, localData);

    // Interpolate
    DeviceBuffer<float> fieldRawData (resolution.x * resolution.y * resolution.z);

    dim3 threads(8, 8, 8);
    dim3 blocks((resolution.x+threads.x-1) / threads.x,
                (resolution.y+threads.y-1) / threads.y,
                (resolution.z+threads.z-1) / threads.z);

    localData.uploadToDevice(0);
    SAFE_KERNEL_LAUNCH(
            InterpolateKernels::cubicInterpolate3D,
            blocks, threads, 0, 0,
            localData.devPtr(), resolutionBeforeInterpolation, initialSdfH,
            fieldRawData.devPtr(), resolution, h, offset, lenScalingFactor );

    setupArrayTexture(fieldRawData.devPtr());
}

void Field::setupArrayTexture(const float *fieldDevPtr)
{
    // Prepare array to be transformed into texture
    auto chDesc = cudaCreateChannelDesc<float>();
    CUDA_Check( cudaMalloc3DArray(&fieldArray, &chDesc, make_cudaExtent(resolution.x, resolution.y, resolution.z)) );

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)fieldDevPtr, resolution.x*sizeof(float), resolution.x, resolution.y);
    copyParams.dstArray = fieldArray;
    copyParams.extent   = make_cudaExtent(resolution.x, resolution.y, resolution.z);
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    CUDA_Check( cudaMemcpy3D(&copyParams) );

    // Create texture
    cudaResourceDesc resDesc = {};
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = fieldArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.addressMode[2]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModePoint;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    CUDA_Check( cudaCreateTextureObject(&fieldTex, &resDesc, &texDesc, nullptr) );

    CUDA_Check( cudaDeviceSynchronize() );
}
