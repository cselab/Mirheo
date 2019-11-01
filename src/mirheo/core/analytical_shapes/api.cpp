#include "api.h"

#define DECLARE_DESC(Shape) const char *Shape::desc = #Shape;

ASHAPE_TABLE(DECLARE_DESC)

