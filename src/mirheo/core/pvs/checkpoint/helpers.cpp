#include "helpers.h"

#include <mirheo/core/xdmf/type_map.h>
#include <mirheo/core/xdmf/xdmf.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/type_shift.h>
#include <mirheo/core/containers.h>

namespace mirheo
{

namespace checkpoint_helpers
{
std::tuple<std::vector<real3>,
           std::vector<real3>,
           std::vector<int64_t>>
splitAndShiftPosVel(const DomainInfo &domain,
                    const PinnedBuffer<real4>& pos4,
                    const PinnedBuffer<real4>& vel4)
{
    auto n = pos4.size();
    std::vector<real3> pos(n), vel(n);
    std::vector<int64_t> ids(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto p = Particle(pos4[i], vel4[i]);
        pos[i] = domain.local2global(p.r);
        vel[i] = p.u;
        ids[i] = p.getId();
    }
    return {std::move(pos), std::move(vel), std::move(ids)};
}

std::tuple<std::vector<real3>, std::vector<RigidReal4>,
           std::vector<RigidReal3>, std::vector<RigidReal3>,
           std::vector<RigidReal3>, std::vector<RigidReal3>>
splitAndShiftMotions(DomainInfo domain, const PinnedBuffer<RigidMotion>& motions)
{
    const size_t n = motions.size();
    std::vector<real3> pos(n);
    std::vector<RigidReal4> quaternion(n);
    std::vector<RigidReal3> vel(n), omega(n), force(n), torque(n);

    for (size_t i = 0; i < n; ++i)
    {
        auto m = motions[i];
        pos[i] = domain.local2global(make_real3(m.r));
        quaternion[i] = static_cast<RigidReal4>(m.q);
        vel[i] = m.vel;
        omega[i] = m.omega;
        force[i] = m.force;
        torque[i] = m.torque;
    }

    return {std::move(pos),
            std::move(quaternion),
            std::move(vel),
            std::move(omega),
            std::move(force),
            std::move(torque)};
}


template<typename Container>
static void shiftElementsLocal2Global(Container& data, const DomainInfo domain)
{
    auto shift = domain.local2global({0._r, 0._r, 0._r});
    for (auto& d : data) type_shift::apply(d, shift);
}

std::vector<XDMF::Channel> extractShiftPersistentData(const DomainInfo& domain,
                                                      const DataManager& extraData,
                                                      const std::set<std::string>& blackList)
{
    std::vector<XDMF::Channel> channels;

    for (auto& namedChannelDesc : extraData.getSortedChannels())
    {
        auto channelName = namedChannelDesc.first;
        auto channelDesc = namedChannelDesc.second;

        if (channelDesc->persistence != DataManager::PersistenceMode::Active)
            continue;

        if (blackList.find(channelName) != blackList.end())
            continue;

        mpark::visit([&](auto bufferPtr)
        {
            using T = typename std::remove_pointer<decltype(bufferPtr)>::type::value_type;
            bufferPtr->downloadFromDevice(defaultStream, ContainersSynch::Synch);

            auto needShift = XDMF::Channel::NeedShift::False;

            if (channelDesc->needShift())
            {
                shiftElementsLocal2Global(*bufferPtr, domain);
                needShift = XDMF::Channel::NeedShift::True;
            }

            auto formtype   = XDMF::getDataForm<T>();
            auto numbertype = XDMF::getNumberType<T>();
            auto datatype   = DataTypeWrapper<T>();

            channels.push_back(XDMF::Channel {channelName, bufferPtr->data(),
                                              formtype, numbertype, datatype, needShift});
        }, channelDesc->varDataPtr);
    }

    return channels;
}

} // namespace checkpoint_helpers

} // namespace mirheo
