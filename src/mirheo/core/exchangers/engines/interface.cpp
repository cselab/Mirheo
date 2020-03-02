#include "interface.h"

#include <mirheo/core/exchangers/exchanger_interfaces.h>

namespace mirheo
{

ExchangeEngine::ExchangeEngine(std::unique_ptr<Exchanger>&& exchanger) :
    exchanger_(std::move(exchanger))
{}

ExchangeEngine::~ExchangeEngine() = default;

} // namespace mirheo
