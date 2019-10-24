#include "obj_rod_binding.h"

#include <core/pvs/views/rov.h>
#include <core/pvs/rigid_object_vector.h>
#include <core/pvs/rod_vector.h>
#include <core/rigid/utils.h>
#include <core/utils/quaternion.h>
#include <core/utils/kernel_launch.h>

namespace ObjRodBindingKernels
{

struct BindingParams
{
    float3 relPos;
    float T;
    float kb;
};

template <typename View>
__device__ inline float3 fetchPosition(View view, int i)
{
    Float3_int ri(view.readPosition(i));
    return make_float3(ri.v);
}

__device__ inline float3 safeNormalize(float3 u)
{
    constexpr float tol = 1e-6f;
    auto nrm2 = dot(u,u);
    if (nrm2 < tol) return make_float3(0.f);
    return math::rsqrt(nrm2) * u;
}

__device__ void applyBindingForce(const DomainInfo& domain,
                                  int i, const ROVview& objs,
                                  int j, const OVview& rods,
                                  const BindingParams& params)
{
    const auto motion = toSingleMotion(objs.motions[i]);
    const float3 relAnchor = Quaternion::rotate( params.relPos, motion.q );
    const float3 anchor = motion.r + relAnchor;

    const int start = j * rods.objSize; 
    auto r0 = fetchPosition(rods, start + 0);
    auto u0 = fetchPosition(rods, start + 1);
    auto u1 = fetchPosition(rods, start + 2);
    auto r1 = fetchPosition(rods, start + 5);

    auto dr = anchor - r0;

    // avoid computing the forces with periodic images
    if (math::abs(dr.x) > 0.5f * domain.localSize.x) return;
    if (math::abs(dr.y) > 0.5f * domain.localSize.y) return;
    if (math::abs(dr.z) > 0.5f * domain.localSize.z) return;

    const auto fanchor = -params.kb * safeNormalize(dr);
    const auto e0 = normalize(r1 - r0);
    auto dp = u1 - u0;
    dp = normalize(dp - e0 * dot(e0, dp));

    float3 T       = params.T * e0;
    float3 fu0     = 0.5f * cross(T, dp);
    float3 Tanchor = cross(relAnchor, fanchor);
    
    atomicAdd(&rods.forces[start + 0], -fanchor);
    atomicAdd(&rods.forces[start + 1],  fu0);
    atomicAdd(&rods.forces[start + 2], -fu0);

    atomicAdd(&objs.motions[i].force , make_rigidReal3(fanchor));
    atomicAdd(&objs.motions[i].torque, make_rigidReal3(Tanchor + T));
}

__global__ void computeBindingForces(DomainInfo domain, ROVview objs, OVview rods, BindingParams params)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    int idObj;
    if (i < objs.nObjects)
        idObj = objs.ids[i];

    extern __shared__ int idRods[];

    for (int j = threadIdx.x; j < rods.nObjects; j += blockDim.x)
        idRods[j] = rods.ids[j];

    __syncthreads();

    if (i >= objs.nObjects) return;

    // HACK; TODO: better approach?
    // there can be several times the same rod because of halo exchanges on a single node (up to 7)
    // we store all indices and then compute forces
    // forces are skipped internally if the distance is more than half subdomain size
    int rodLocs[7];
    int nFound = 0;

    for (int j = 0; j < rods.nObjects; ++j)
        if (idRods[j] == idObj)
            rodLocs[nFound++] = j;

    for (int j = 0; j < nFound; ++j)
        applyBindingForce(domain, i, objs, rodLocs[j], rods, params);
}

} // namespace ObjRodBindingKernels


ObjectRodBindingInteraction::ObjectRodBindingInteraction(const MirState *state, std::string name,
                                                         float torque, float3 relAnchor, float kBound) :
    Interaction(state, name, /*rc*/ 1.0f),
    torque(torque),
    relAnchor(relAnchor),
    kBound(kBound)
{}

ObjectRodBindingInteraction::~ObjectRodBindingInteraction() = default;

void ObjectRodBindingInteraction::setPrerequisites(__UNUSED ParticleVector *pv1,
                                                   __UNUSED ParticleVector *pv2,
                                                   __UNUSED CellList *cl1,
                                                   __UNUSED CellList *cl2)
{}

void ObjectRodBindingInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                        __UNUSED CellList *cl1, __UNUSED CellList *cl2,
                                        cudaStream_t stream)
{
    auto rov1 = dynamic_cast<RigidObjectVector*>(pv1);
    auto rv1  = dynamic_cast<RodVector*>(pv1);

    auto rov2 = dynamic_cast<RigidObjectVector*>(pv2);
    auto rv2  = dynamic_cast<RodVector*>(pv2);

    if      ((rov1 != nullptr) && (rv2 != nullptr)) return _local(rov1, rv2, stream);
    else if ((rov2 != nullptr) && (rv1 != nullptr)) return _local(rov2, rv1, stream);

    die("Local interactions '%s' must be given one RigidObjectVector and one RodVector", name.c_str());
}

void ObjectRodBindingInteraction::halo(ParticleVector *pv1, ParticleVector *pv2,
                                       __UNUSED CellList *cl1, __UNUSED CellList *cl2, cudaStream_t stream)
{
    auto rov1 = dynamic_cast<RigidObjectVector*>(pv1);
    auto rv1  = dynamic_cast<RodVector*>(pv1);

    auto rov2 = dynamic_cast<RigidObjectVector*>(pv2);
    auto rv2  = dynamic_cast<RodVector*>(pv2);

    if      ((rov1 != nullptr) && (rv2 != nullptr)) return _halo(rov1, rv2, stream);
    else if ((rov2 != nullptr) && (rv1 != nullptr)) return _halo(rov2, rv1, stream);

    die("Local interactions '%s' must be given one RigidObjectVector and one RodVector", name.c_str());    
}


void ObjectRodBindingInteraction::_local(RigidObjectVector *rov, RodVector *rv, cudaStream_t stream) const
{
    ROVview objs(rov, rov->local());
    OVview  rods(rv,   rv->local());

    const int nthreads = 64;
    const int nblocks  = getNblocks(objs.nObjects, nthreads);
    const size_t shMem = rods.nObjects * sizeof(int);
    
    ObjRodBindingKernels::BindingParams params {relAnchor, torque, kBound};
    
    SAFE_KERNEL_LAUNCH(
        ObjRodBindingKernels::computeBindingForces,
        nblocks, nthreads, shMem, stream,
        state->domain, objs, rods, params);
}

void ObjectRodBindingInteraction::_halo(RigidObjectVector *rov, RodVector *rv, cudaStream_t stream) const
{
    ROVview objs(rov, rov->local());
    OVview  rods(rv,   rv->halo());

    const int nthreads = 64;
    const int nblocks  = getNblocks(objs.nObjects, nthreads);
    const size_t shMem = rods.nObjects * sizeof(int);
    
    ObjRodBindingKernels::BindingParams params {relAnchor, torque, kBound};
    
    SAFE_KERNEL_LAUNCH(
        ObjRodBindingKernels::computeBindingForces,
        nblocks, nthreads, shMem, stream,
        state->domain, objs, rods, params);
}
