// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "interface.h"

#include <mirheo/core/exchangers/interface.h>

namespace mirheo
{

ExchangeEngine::ExchangeEngine(std::unique_ptr<Exchanger>&& exchanger) :
    exchanger_(std::move(exchanger))
{}

ExchangeEngine::~ExchangeEngine() = default;

} // namespace mirheo
