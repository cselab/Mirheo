// Copyright 2020 ETH Zurich. All Rights Reserved.
#pragma once

#include "kernels/type_traits.h"

#include <array>

#include <mirheo/core/datatypes.h>  //real3
#include <mirheo/core/celllist.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/pvs/views/pv.h>

#include <cassert>
#include <type_traits>

namespace mirheo
{

///Template parameter for whether we need to add force to dst(Self) or src(Other)
enum class InteractionType
{
    LLL,
    HLL,
    LHH,
};

//TODO: should be given as a boolean if StressTensorPlugin is used!
constexpr bool ComputeHaloForces = false;   //this is used for calculating stress tensor

/** \brief Compute triplewise interactions within a single ParticleVector.
    \tparam Interaction The triplewise interaction kernel

    \param [in] cinfo cell-list data
    \param [in,out] view The view that contains the particle data
    \param [in] interaction The triplewise interaction kernel

    Mapping is one thread per particle.
    TODO: Explain the algorithm.
  */
template <InteractionType InteractType, typename Handler>
// __launch_bounds__(128, 16)
__device__ void computeTriplewiseSelfInteractions(
        CellListInfo cinfo, typename Handler::ViewType dstView, typename Handler::ViewType srcView, Handler handler)
{
    const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
    if (dstId >= dstView.size)
        return;

    const auto dstP = handler.read(dstView, dstId);

    real3 frc_ = make_real3(0.0_r);

    real4 * __restrict__ srcForces = srcView.forces;
    
    typename Handler::ParticleType srcP1, srcP2;

    const int3 cell0 = cinfo.getCellIdAlongAxes(handler.getPosition(dstP));
    
    constexpr int padding = InteractType == InteractionType::LHH ? 2 : 1;
    const int cellZMin = math::max(cell0.z - padding, 0);
    const int cellZMax = math::min(cell0.z + padding, cinfo.ncells.z - 1);
    const int cellYMin = math::max(cell0.y - padding, 0);
    const int cellYMax = math::min(cell0.y + padding, cinfo.ncells.y - 1);
    const int cellXMin = math::max(cell0.x - padding, 0);
    const int cellXMax = math::min(cell0.x + padding + 1, cinfo.ncells.x);

    for (int cellZ1 = cellZMin; cellZ1 <= cellZMax; ++cellZ1)
    {
        for (int cellY1 = cellYMin; cellY1 <= cellYMax; ++cellY1)
        {
            const int rowStart1 = cinfo.encode(cellXMin, cellY1, cellZ1);
            const int rowEnd1 = cinfo.encode(cellXMax, cellY1, cellZ1);

            const int pstart1 = cinfo.cellStarts[rowStart1];
            const int pend1   = cinfo.cellStarts[rowEnd1];

            for (int cellZ2 = cellZ1; cellZ2 <= cellZMax; ++cellZ2)
            {
                for (int cellY2 = (cellZ1 == cellZ2) ? cellY1 : cellYMin; cellY2 <= cellYMax; ++cellY2)
                {
                    const int rowStart2 = cinfo.encode(cellXMin, cellY2, cellZ2);
                    const int rowEnd2 = cinfo.encode(cellXMax, cellY2, cellZ2);

                    const int pstart2 = cinfo.cellStarts[rowStart2];
                    const int pend2   = cinfo.cellStarts[rowEnd2];
                    
                    for (int srcId1 = pstart1; srcId1 < pend1; ++srcId1)
                    {
                        handler.readCoordinates(srcP1, srcView, srcId1);
                        const bool interacting01 = handler.withinCutoff(dstP, srcP1);
                        real3 force1 = make_real3(0.0_r);
                        for (int srcId2 = (cellZ2 == cellZ1) && (cellY2 == cellY1) ? srcId1 + 1 : pstart2; srcId2 < pend2; ++srcId2)
                        {
                            if (InteractType == InteractionType::LLL && (dstId == srcId1 || dstId == srcId2)) continue;

                            handler.readCoordinates(srcP2, srcView, srcId2);

                            const bool interacting20 = handler.withinCutoff(dstP , srcP2);
                            const bool interacting12 = handler.withinCutoff(srcP1, srcP2);

                            bool condition;
                            if (InteractType == InteractionType::LLL) {
                                condition = interacting01 && interacting20 && (
                                        !interacting12
                                        || (dstId < srcId1 && dstId < srcId2));
                            } else { // HLL or LHH
                                condition = (interacting01 && interacting12)
                                         || (interacting12 && interacting20)
                                         || (interacting20 && interacting01);
                            }

                            if (condition) {
                                handler.readExtraData(srcP1, srcView, srcId1);
                                handler.readExtraData(srcP2, srcView, srcId2);

                                const std::array<real3, 3> val = handler(dstP, srcP1, srcP2, interacting01, interacting12, interacting20);
                                if (InteractType != InteractionType::HLL || ComputeHaloForces)
                                    frc_ += val[0];
                                if (InteractType != InteractionType::LHH || ComputeHaloForces) {
                                    force1 += val[1];
                                    atomicAdd(srcForces + srcId2, val[2]);
                                }
                            }
                        }
                        if (InteractType != InteractionType::LHH || ComputeHaloForces)
                            atomicAdd(srcForces + srcId1, force1);
                    }
                } //cellY2
            } //cellZ2
        } //cellY1
    } //cellZ1
    if (InteractType != InteractionType::HLL || ComputeHaloForces)
        atomicAdd(dstView.forces + dstId, frc_);
}

//Debugging purpose (nvvp)
template <typename Handler>
__global__ void LLL_(CellListInfo cinfo, typename Handler::ViewType dstView, typename Handler::ViewType srcView, Handler handler){
    computeTriplewiseSelfInteractions<InteractionType::LLL>(cinfo, dstView, srcView, handler);
}

template <typename Handler>
__global__ void LHH_(CellListInfo cinfo, typename Handler::ViewType dstView, typename Handler::ViewType srcView, Handler handler){
    computeTriplewiseSelfInteractions<InteractionType::LHH>(cinfo, dstView, srcView, handler);
}

template <typename Handler>
__global__ void HLL_(CellListInfo cinfo, typename Handler::ViewType dstView, typename Handler::ViewType srcView, Handler handler){
    computeTriplewiseSelfInteractions<InteractionType::HLL>(cinfo, dstView, srcView, handler);
}

} // namespace mirheo
