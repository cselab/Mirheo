// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "rdf.h"

#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/file_wrapper.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>
#include <mirheo/core/utils/path.h>

namespace mirheo
{

namespace rdf_plugin_kernels
{
using rdf_plugin::CountType;

__global__ void getCountsPerDistance(PVview view, CellListInfo cinfo, CountType *totCounts, int nbins, real maxDist, CountType *totNparts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int nparts;
    extern __shared__ int counts[];

    if (threadIdx.x == 0)
        nparts = 0;

    for (int k = threadIdx.x; k < nbins; k += blockDim.x)
        counts[k] = 0;

    __syncthreads();

    if (i < view.size)
    {
        const real3 ri = Real3_int(view.readPosition(i)).v;

        const real3 L = cinfo.localDomainSize;

        const real invh = nbins / maxDist;

        // skip inner halo particles for simplicity
        if (ri.x > -0.5_r * L.x + maxDist &&
            ri.y > -0.5_r * L.y + maxDist &&
            ri.z > -0.5_r * L.z + maxDist &&
            ri.x <= 0.5_r * L.x - maxDist &&
            ri.y <= 0.5_r * L.y - maxDist &&
            ri.z <= 0.5_r * L.z - maxDist)
        {
            atomicAdd(&nparts, 1);

            const int3 c0 = cinfo.getCellIdAlongAxes(ri);
            int3 c1;
            for (c1.z = math::max(c0.z-1, 0); c1.z <= math::min(c0.z+1, cinfo.ncells.z-1); ++c1.z) {
                for (c1.y = math::max(c0.y-1, 0); c1.y <= math::min(c0.y+1, cinfo.ncells.y-1); ++c1.y) {
                    for (c1.x = math::max(c0.x-1, 0); c1.x <= math::min(c0.x+1, cinfo.ncells.x-1); ++c1.x) {
                        const int cid1 = cinfo.encode(c1);
                        const int start = cinfo.cellStarts[cid1];
                        const int end   = cinfo.cellStarts[cid1+1];

                        for (int j = start; j < end; ++j)
                        {
                            if (j == i)
                                continue;

                            const real3 rj = Real3_int(view.readPosition(j)).v;
                            const real r = length(rj - ri);

                            const int k = r * invh;
                            if (k < nbins)
                                atomicAdd(&counts[k], 1);
                        }
                    }
                }
            }
        }
    }

    // accumulate to global memory
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(totNparts, nparts);

    for (int k = threadIdx.x; k < nbins; k += blockDim.x)
        atomicAdd(&totCounts[k], counts[k]);
}

} // namespace rdf_plugin_kernels

RdfPlugin::RdfPlugin(const MirState *state, std::string name, std::string pvName, real maxDist, int nbins, int computeEvery) :
    SimulationPlugin(state, name),
    pvName_(pvName),
    maxDist_(maxDist),
    nbins_(nbins),
    computeEvery_(computeEvery)
{
    countsPerBin_.resize_anew(nbins);
}

RdfPlugin::RdfPlugin(const MirState *state, Loader&, const ConfigObject& config)
    : RdfPlugin(state, config["name"], config["pvName"], config["maxDist"],
                config["nbins"], config["computeEvery"])
{}

RdfPlugin::~RdfPlugin() = default;

void RdfPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);
    pv_ = simulation->getPVbyNameOrDie(pvName_);
    cl_ = std::make_unique<CellList>(pv_, maxDist_, getState()->domain.localSize);
}

void RdfPlugin::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), computeEvery_))
        return;

    countsPerBin_.clear(stream);
    nparticles_.clear(stream);

    cl_->build(stream);

    auto view = cl_->getView<PVview>();
    constexpr int nthreads = 128;
    const size_t smem = nbins_ * sizeof(int);

    SAFE_KERNEL_LAUNCH(
                rdf_plugin_kernels::getCountsPerDistance,
                getNblocks(view.size, nthreads), nthreads, smem, stream,
                view, cl_->cellInfo(), countsPerBin_.devPtr(), nbins_, maxDist_, nparticles_.devPtr() );

    countsPerBin_.downloadFromDevice(stream, ContainersSynch::Asynch);
    nparticles_  .downloadFromDevice(stream);

    needToDump_ = true;
}

void RdfPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (needToDump_) {
        _waitPrevSend();

        const real3 L = getState()->domain.localSize;
        const real V = (L.x - 2 * maxDist_) * (L.y - 2 * maxDist_) * (L.z - 2 * maxDist_);

        const MirState::StepType timeStamp = getTimeStamp(getState(), computeEvery_);
        SimpleSerializer::serialize(sendBuffer_, timeStamp, nparticles_[0],
                                    countsPerBin_, maxDist_, V);
        _send(sendBuffer_);
        needToDump_ = false;
    }
}

void RdfPlugin::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<RdfPlugin>(this, _saveSnapshot(saver, "RdfPlugin"));
}

ConfigObject RdfPlugin::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = SimulationPlugin::_saveSnapshot(saver, typeName);
    config.emplace("pvName", saver(pvName_));
    config.emplace("maxDist", saver(maxDist_));
    config.emplace("nbins", saver(nbins_));
    config.emplace("computeEvery", saver(computeEvery_));
    return config;
}



RdfDump::RdfDump(std::string name, std::string basename) :
    PostprocessPlugin(name),
    basename_(std::move(basename))
{}

RdfDump::RdfDump(Loader&, const ConfigObject& config)
    : RdfDump(config["name"], config["basename"])
{}

void RdfDump::setup(const MPI_Comm& comm, const MPI_Comm& interComm)
{
    PostprocessPlugin::setup(comm, interComm);
    createFoldersCollective(comm, getParentPath(basename_));
}

void RdfDump::deserialize()
{
    MirState::StepType timeStamp;
    rdf_plugin::CountType nparticles;
    std::vector<rdf_plugin::CountType> countsPerBin;
    real maxDist;
    real V;

    SimpleSerializer::deserialize(data_, timeStamp, nparticles, countsPerBin, maxDist, V);

    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : &nparticles, &nparticles, 1, getMPIIntType<rdf_plugin::CountType>(), MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : &V, &V, 1, getMPIFloatType<real>(), MPI_SUM, 0, comm_) );

    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : countsPerBin.data(), countsPerBin.data(), countsPerBin.size(),
                          getMPIIntType<rdf_plugin::CountType>(), MPI_SUM, 0, comm_) );

    if (rank_ == 0)
    {
        const int nbins = static_cast<int>(countsPerBin.size());
        const real h = maxDist / nbins;

        const real numDensity = nparticles / V;

        const std::string fname = basename_ + createStrZeroPadded(timeStamp) + ".csv";
        FileWrapper f(fname, "w");

        fprintf(f.get(), "r,rdf\n");

        for (int k = 0; k < nbins; ++k)
        {
            const double r0 = h * k;
            const double r1 = r0 + h;
            const double r = 0.5 * (r0+r1);
            const double shellVolume = 4.0 * M_PI / 3.0 * (r1*r1*r1 - r0*r0*r0);
            const double g = countsPerBin[k] / (shellVolume * nparticles * numDensity);

            fprintf(f.get(), "%g,%g\n", r, g);
        }
    }
}

void RdfDump::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<RdfDump>(this, _saveSnapshot(saver, "RdfDump"));
}

ConfigObject RdfDump::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = PostprocessPlugin::_saveSnapshot(saver, typeName);
    config.emplace("basename", saver(basename_));
    return config;
}

} // namespace mirheo
