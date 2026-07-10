// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "factory_helper.h"

namespace mirheo
{

namespace factory_helper
{

SW3Params readSW3Params(ParametersWrap &desc)
{
    SW3Params p;
    p.lambda = desc.read<real>("lambda_");
    p.epsilon = desc.read<real>("epsilon");
    p.theta = desc.read<real>("theta");
    p.gamma = desc.read<real>("gamma");
    p.sigma = desc.read<real>("sigma");
    return p;
}

DummyParams readDummyParams(ParametersWrap& desc)
{
    DummyParams p;
    p.epsilon = desc.read<real>("epsilon");
    return p;
}


} // namespace factory_helper

} // namespace mirheo
