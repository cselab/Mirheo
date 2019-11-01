#include "api.h"

#define DECLARE_DESC(Shape) const char *Shape::desc = #Shape;

namespace mirheo
{

ASHAPE_TABLE(DECLARE_DESC)

} // namespace mirheo
