#pragma once

#include "kernels/type_traits.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/pvs/views/pv.h>

#include <cassert>
#include <type_traits>

namespace mirheo
{

/// Used as template parameter to differentiate self interaction (src and dst are the same pvs)
/// from extrenal interactions (src and dst are different pvs)
enum class InteractionWith
{
    Self, Other
};

/// Used as template parameter to state if the interaction must save its output or not
enum class InteractionOutMode
{
    NeedOutput,
    NoOutput
};

/// Template parameter that controls how the particles are fetched
/// (performance related)
enum class InteractionFetchMode
{
    RowWise, ///< fetched cell-row by cell-row (better for e.g. densely mixed particles)
    Dilute   ///< fetched cell by cell (better for e.g. halo interactions)
};

/**  Compute interactions between one destination particle and
     all source particles in a given cell, defined by range of ids [pstart, pend).

     \tparam NeedDstOutput if Set to NeedOutput, the force/density/stress of the dst pv will be updated
     \tparam NeedSrcOutput if Set to NeedOutput, the force/density/stress of the src pv will be updated
     \tparam InteractWith states if it is a self interaction or not
     \tparam Interaction The pairwise kernel
     \tparam Accumulator Used to accumulate the output of the kernel

     \param [in] pstart lower bound of id range of the particles to be worked on (inclusive)
     \param [in] pend  upper bound of id range (exclusive)
     \param [in] dstP destination particle
     \param [in] dstId destination particle local index
     \param [in,out] srcView The view of the src particle vector
     \param [in] interaction The pairwise interaction kernel
     \param [in,out] accumulator Manages the accumulated output on the dst particle
 */
template<InteractionOutMode NeedDstOutput, InteractionOutMode NeedSrcOutput, InteractionWith InteractWith,
         typename Interaction, typename Accumulator>
__device__ inline void computeCell(
        int pstart, int pend,
        typename Interaction::ParticleType dstP, int dstId, typename Interaction::ViewType srcView,
        Interaction& interaction, Accumulator& accumulator)
{
    for (int srcId = pstart; srcId < pend; srcId++)
    {
        typename Interaction::ParticleType srcP;
        interaction.readCoordinates(srcP, srcView, srcId);

        bool interacting = interaction.withinCutoff(srcP, dstP);

        if (InteractWith == InteractionWith::Self)
            if (dstId <= srcId)
                interacting = false;

        if (interacting)
        {
            interaction.readExtraData(srcP, srcView, srcId);

            const auto val = interaction(dstP, dstId, srcP, srcId);

            if (NeedDstOutput == InteractionOutMode::NeedOutput)
                accumulator.add(val);

            if (NeedSrcOutput == InteractionOutMode::NeedOutput)
                accumulator.atomicAddToSrc(val, srcView, srcId);
        }
    }
}

/** \brief Compute interactions within a single ParticleVector.
    \tparam Interaction The pairwise interaction kernel

    \param [in] cinfo cell-list data
    \param [in,out] view The view that contains the particle data
    \param [in] interaction The pairwise interaction kernel

    Mapping is one thread per particle. The thread will traverse half
    of the neighbouring cells and compute all the interactions between
    the destination particle and all the particles in the cells.
 */
template<typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeSelfInteractions(
        CellListInfo cinfo, typename Interaction::ViewType view, Interaction interaction)
{
    const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
    if (dstId >= view.size) return;

    const auto dstP = interaction.read(view, dstId);

    auto accumulator = interaction.getZeroedAccumulator();

    const int3 cell0 = cinfo.getCellIdAlongAxes(interaction.getPosition(dstP));

    for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
    {
        for (int cellY = cell0.y-1; cellY <= cell0.y; cellY++)
        {
            if ( !(cellY >= 0 && cellY < cinfo.ncells.y && cellZ >= 0 && cellZ < cinfo.ncells.z) ) continue;
            if (cellY == cell0.y && cellZ > cell0.z) continue;

            const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
            const int rowStart  = math::max(midCellId-1, 0);
            int rowEnd          = math::min(midCellId+2, cinfo.totcells);

            if ( cellY == cell0.y && cellZ == cell0.z ) rowEnd = midCellId + 1; // this row is already partly covered

            const int pstart = cinfo.cellStarts[rowStart];
            const int pend   = cinfo.cellStarts[rowEnd];

            if (cellY == cell0.y && cellZ == cell0.z)
                computeCell<InteractionOutMode::NeedOutput, InteractionOutMode::NeedOutput, InteractionWith::Self>
                    (pstart, pend, dstP, dstId, view, interaction, accumulator);
            else
                computeCell<InteractionOutMode::NeedOutput, InteractionOutMode::NeedOutput, InteractionWith::Other>
                    (pstart, pend, dstP, dstId, view, interaction, accumulator);
        }
    }

    if (needSelfInteraction<Interaction>::value)
        accumulator.add(interaction(dstP, dstId, dstP, dstId));

    accumulator.atomicAddToDst(accumulator.get(), view, dstId);
}


/** \brief Compute the interactions between particle of two different ParticleVector.
    \tparam NeedDstOutput States if the dstination particles must be modified
    \tparam NeedSrcOutput States if the source particles must be modified
    \tparam FetchMode Performance parameter; controls how to traverse the src particles
    \tparam Interaction The pairwise interaction kernel

    \param [in,out] dstView Destination particles data
    \param [in] srcCinfo Cell-lists info of the source particles
    \param [in,out] srcView Source particles data
    \param [in] interaction Instance of the pairwise kernel functor

    Mapping is one thread per destination particle.
    The thread will traverse all of the neighbouring cells and compute the interactions between
    the destination particle and all the particles of the \p srcView in those cells.
 */
template<InteractionOutMode NeedDstOutput, InteractionOutMode NeedSrcOutput, InteractionFetchMode FetchMode, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_1tpp(
        typename Interaction::ViewType dstView, CellListInfo srcCinfo,
        typename Interaction::ViewType srcView, Interaction interaction)
{
    static_assert(NeedDstOutput == InteractionOutMode::NeedOutput || NeedSrcOutput == InteractionOutMode::NeedOutput,
                  "External interactions should return at least one output");

    const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
    if (dstId >= dstView.size) return;

    const auto dstP = interaction.readNoCache(dstView, dstId);

    auto accumulator = interaction.getZeroedAccumulator();

    const int3 cell0 = srcCinfo.getCellIdAlongAxes<CellListsProjection::NoClamp>(interaction.getPosition(dstP));

    for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
        for (int cellY = cell0.y-1; cellY <= cell0.y+1; cellY++)
            if (FetchMode == InteractionFetchMode::RowWise)
            {
                if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

                const int midCellId = srcCinfo.encode(cell0.x, cellY, cellZ);
                const int rowStart  = math::max(midCellId-1, 0);
                const int rowEnd    = math::min(midCellId+2, srcCinfo.totcells);

                if (rowStart >= rowEnd) continue;

                const int pstart = srcCinfo.cellStarts[rowStart];
                const int pend   = srcCinfo.cellStarts[rowEnd];

                computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
                    (pstart, pend, dstP, dstId, srcView, interaction, accumulator);
            }
            else
            {
                if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

                for (int cellX = math::max(cell0.x-1, 0); cellX <= math::min(cell0.x+1, srcCinfo.ncells.x-1); cellX++)
                {
                    const int cid = srcCinfo.encode(cellX, cellY, cellZ);
                    const int pstart = srcCinfo.cellStarts[cid];
                    const int pend   = srcCinfo.cellStarts[cid+1];

                    computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
                        (pstart, pend, dstP, dstId, srcView, interaction, accumulator);
                }
            }

    if (NeedDstOutput == InteractionOutMode::NeedOutput)
        accumulator.atomicAddToDst(accumulator.get(), dstView, dstId);
}

/** Same as computeExternalInteractions_1tpp()
    With a mapping of 3 threads per destination particle (one per adjacent cell plane).
    Used to increase parallelization when the number of particles is lower.
 */
template<InteractionOutMode NeedDstOutput, InteractionOutMode NeedSrcOutput, InteractionFetchMode FetchMode, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_3tpp(
        typename Interaction::ViewType dstView, CellListInfo srcCinfo,
        typename Interaction::ViewType srcView, Interaction interaction)
{
    static_assert(NeedDstOutput == InteractionOutMode::NeedOutput || NeedSrcOutput == InteractionOutMode::NeedOutput,
                  "External interactions should return at least one output");

    const int gid = blockIdx.x*blockDim.x + threadIdx.x;

    const int dstId = gid / 3;
    const int dircode = gid % 3 - 1;

    if (dstId >= dstView.size) return;

    const auto dstP = interaction.readNoCache(dstView, dstId);

    auto accumulator = interaction.getZeroedAccumulator();

    const int3 cell0 = srcCinfo.getCellIdAlongAxes<CellListsProjection::NoClamp>(interaction.getPosition(dstP));

    int cellZ = cell0.z + dircode;

    for (int cellY = cell0.y-1; cellY <= cell0.y+1; cellY++)
    {
        if (FetchMode == InteractionFetchMode::RowWise)
        {
            if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

            const int midCellId = srcCinfo.encode(cell0.x, cellY, cellZ);
            const int rowStart  = math::max(midCellId-1, 0);
            const int rowEnd    = math::min(midCellId+2, srcCinfo.totcells);

            if (rowStart >= rowEnd) continue;

            const int pstart = srcCinfo.cellStarts[rowStart];
            const int pend   = srcCinfo.cellStarts[rowEnd];

            computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
                (pstart, pend, dstP, dstId, srcView, interaction, accumulator);
        }
        else
        {
            if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) continue;

            for (int cellX = math::max(cell0.x-1, 0); cellX <= math::min(cell0.x+1, srcCinfo.ncells.x-1); cellX++)
            {
                const int cid = srcCinfo.encode(cellX, cellY, cellZ);
                const int pstart = srcCinfo.cellStarts[cid];
                const int pend   = srcCinfo.cellStarts[cid+1];

                computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
                    (pstart, pend, dstP, dstId, srcView, interaction, accumulator);
            }
        }
    }

    if (NeedDstOutput == InteractionOutMode::NeedOutput)
        accumulator.atomicAddToDst(accumulator.get(), dstView, dstId);
}

/** Same as computeExternalInteractions_1tpp()
    With a mapping of 9 threads per destination particle (one per adjacent cell row).
    Used to increase parallelization when the number of particles is lower.
 */
template<InteractionOutMode NeedDstOutput, InteractionOutMode NeedSrcOutput, InteractionFetchMode FetchMode, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_9tpp(
        typename Interaction::ViewType dstView, CellListInfo srcCinfo,
        typename Interaction::ViewType srcView, Interaction interaction)
{
    static_assert(NeedDstOutput == InteractionOutMode::NeedOutput || NeedSrcOutput == InteractionOutMode::NeedOutput,
                  "External interactions should return at least one output");

    const int gid = blockIdx.x*blockDim.x + threadIdx.x;

    const int dstId = gid / 9;
    const int dircode = gid % 9;

    if (dstId >= dstView.size) return;

    const auto dstP = interaction.readNoCache(dstView, dstId);

    auto accumulator = interaction.getZeroedAccumulator();

    const int3 cell0 = srcCinfo.getCellIdAlongAxes<CellListsProjection::NoClamp>(interaction.getPosition(dstP));

    const int cellZ = cell0.z + dircode / 3 - 1;
    const int cellY = cell0.y + dircode % 3 - 1;

    if (FetchMode == InteractionFetchMode::RowWise)
    {
        if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) return;

        const int midCellId = srcCinfo.encode(cell0.x, cellY, cellZ);
        const int rowStart  = math::max(midCellId-1, 0);
        const int rowEnd    = math::min(midCellId+2, srcCinfo.totcells);

        if (rowStart >= rowEnd) return;

        const int pstart = srcCinfo.cellStarts[rowStart];
        const int pend   = srcCinfo.cellStarts[rowEnd];

        computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
            (pstart, pend, dstP, dstId, srcView, interaction, accumulator);
    }
    else
    {
        if ( !(cellY >= 0 && cellY < srcCinfo.ncells.y && cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) return;

        for (int cellX = math::max(cell0.x-1, 0); cellX <= math::min(cell0.x+1, srcCinfo.ncells.x-1); cellX++)
        {
            const int cid = srcCinfo.encode(cellX, cellY, cellZ);
            const int pstart = srcCinfo.cellStarts[cid];
            const int pend   = srcCinfo.cellStarts[cid+1];

            computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
                (pstart, pend, dstP, dstId, srcView, interaction, accumulator);
        }
    }

    if (NeedDstOutput == InteractionOutMode::NeedOutput)
        accumulator.atomicAddToDst(accumulator.get(), dstView, dstId);
}

/** Same as computeExternalInteractions_1tpp()
    With a mapping of 27 threads per destination particle (one per adjacent cell).
    Used to increase parallelization when the number of particles is lower.
 */
template<InteractionOutMode NeedDstOutput, InteractionOutMode NeedSrcOutput, InteractionFetchMode FetchMode, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions_27tpp(
        typename Interaction::ViewType dstView, CellListInfo srcCinfo,
        typename Interaction::ViewType srcView, Interaction interaction)
{
    static_assert(NeedDstOutput == InteractionOutMode::NeedOutput || NeedSrcOutput == InteractionOutMode::NeedOutput,
                  "External interactions should return at least one output");

    const int gid = blockIdx.x*blockDim.x + threadIdx.x;

    const int dstId = gid / 27;
    const int dircode = gid % 27;

    if (dstId >= dstView.size) return;

    const auto dstP = interaction.readNoCache(dstView, dstId);

    auto accumulator = interaction.getZeroedAccumulator();

    const int3 cell0 = srcCinfo.getCellIdAlongAxes<CellListsProjection::NoClamp>(interaction.getPosition(dstP));

    int cellZ = cell0.z +  dircode / 9      - 1;
    int cellY = cell0.y + (dircode / 3) % 3 - 1;
    int cellX = cell0.x +  dircode % 3      - 1;

    if ( !( cellX >= 0 && cellX < srcCinfo.ncells.x &&
            cellY >= 0 && cellY < srcCinfo.ncells.y &&
            cellZ >= 0 && cellZ < srcCinfo.ncells.z) ) return;

    const int cid = srcCinfo.encode(cellX, cellY, cellZ);
    const int pstart = srcCinfo.cellStarts[cid];
    const int pend   = srcCinfo.cellStarts[cid+1];

    computeCell<NeedDstOutput, NeedSrcOutput, InteractionWith::Other>
        (pstart, pend, dstP, dstId, srcView, interaction, accumulator);

    if (NeedDstOutput == InteractionOutMode::NeedOutput)
        accumulator.atomicAddToDst(accumulator.get(), dstView, dstId);
}

} // namespace mirheo
