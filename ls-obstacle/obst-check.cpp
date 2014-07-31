/*
 * obst.cpp
 *
 *  Created on: Jul 30, 2014
 *      Author: kirill
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <cmath>
#include "funnel-obstacle.h"

void checkReadWrite()
{
  std::string inputFileName = "bla.dat";
  FunnelObstacle fo(8.0f);
  fo.write(inputFileName);

  FunnelObstacle fIn(8.0f, inputFileName);
  assert(fIn == fo);
 //to be done
}

void checkFind()
{
  FunnelObstacle fo(8.0f);

  // pick up some points
  const float eps = 0.1;
  const float my0 = -4.0;
  const float yPlaneUp = 4.0;
  assert(fo.sample(0, my0).second < eps);
  assert(fo.isInside(0, my0 + 1) == true);
  assert(fo.isInside(0, my0 - 1) == false);

  // check on the points from the border
  size_t szForEvery = 20;
  float h = 10.0 / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * h - 5;
    float y = 0.0;
    if (x > -1.9 && x <= 1.9)
      assert(fo.isInside(x, y));
    if (x < -2.1 || x >= 2.1)
      assert(!fo.isInside(x, y));
  }

  for (size_t iy = 0; iy < szForEvery; ++iy) {
    float y = iy * h - 5;
    float x = 0.0;
    if (y > -3.9 && y <= 3.9)
      assert(fo.isInside(x, y));
    if (y < -4.1 || y >= 4.1)
      assert(!fo.isInside(x, y));
  }

  /* not accurate for now
  // check on the points from the border
  szForEvery = 20;
  h = 2.0 * sqrtf(yPlaneUp - my0) / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * h - sqrtf(yPlaneUp - my0);
    float y = x*x + my0;
    float dist = fo.sample(x, y).second;
    std::cout << x << ", " << y << " -> "  << dist << std::endl;
    //assert(dist < eps);
  }*/
}

int main()
{
  checkReadWrite();
}
