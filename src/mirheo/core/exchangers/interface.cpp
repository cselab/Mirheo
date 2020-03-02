#include "exchange_entity.h"
#include "interface.h"

namespace mirheo
{

Exchanger::~Exchanger() = default;

void Exchanger::addExchangeEntity(std::unique_ptr<ExchangeEntity>&& e)
{
    helpers_.push_back(std::move(e));
}

ExchangeEntity* Exchanger::getExchangeEntity(size_t id)
{
    return helpers_[id].get();
}

const ExchangeEntity* Exchanger::getExchangeEntity(size_t id) const
{
    return helpers_[id].get();
}

size_t Exchanger::getNumExchangeEntities() const
{
    return helpers_.size();
}

} // namespace mirheo
