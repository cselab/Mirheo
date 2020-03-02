#include "exchange_helpers.h"
#include "exchanger_interfaces.h"

namespace mirheo
{

Exchanger::~Exchanger() = default;

void Exchanger::addExchangeEntity(std::unique_ptr<ExchangeHelper>&& e)
{
    helpers_.push_back(std::move(e));
}

ExchangeHelper* Exchanger::getExchangeEntity(size_t id)
{
    return helpers_[id].get();
}

const ExchangeHelper* Exchanger::getExchangeEntity(size_t id) const
{
    return helpers_[id].get();
}

size_t Exchanger::getNumExchangeEntities() const
{
    return helpers_.size();
}

} // namespace mirheo
