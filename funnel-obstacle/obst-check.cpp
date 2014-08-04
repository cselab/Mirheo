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

#define printOk() do{std::cout << "Quick test " << __func__ << ": OK\n";}while(0)

#define assertTrue(res) do{ if (!(res)) {std::cout << "Quick test " << __func__ << ": FAIL\n";} assert(res);}while(0)

void checkReadWrite()
{
  std::string inputFileName = "bla.dat";
  FunnelObstacle fo(16.0f, 40.0f, 128);
  fo.write(inputFileName);

  FunnelObstacle fIn(16.0f, 40.0f, 128, inputFileName);
  assertTrue(fIn == fo);
  printOk();
 //to be done
}

void checkFind1()
{
  FunnelObstacle fo(8.0f, 10.0f);

  // pick up some points
  const float eps = 0.1;
  const float my0 = -4.0;
  //const float yPlaneUp = 4.0;
  assertTrue(fo.sample(0, my0).second < eps);
  assertTrue(fo.isInside(0, my0 + 1) == true);
  assertTrue(fo.isInside(0, my0 - 1) == false);

  // check on the points from the border
  size_t szForEvery = 20;
  float h = 10.0 / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * h - 5;
    float y = 0.0;
    if (x > -1.9 && x <= 1.9)
      assertTrue(fo.isInside(x, y));
    if (x < -2.1 || x >= 2.1)
      assertTrue(!fo.isInside(x, y));
  }

  for (size_t iy = 0; iy < szForEvery; ++iy) {
    float y = iy * h - 5;
    float x = 0.0;
    if (y > -3.9 && y <= 3.9)
      assertTrue(fo.isInside(x, y));
    if (y < -4.1 || y >= 4.1)
      assertTrue(!fo.isInside(x, y));
  }
  printOk();
}

void checkFind2()
{
  float domainLength = 40.0f;
  FunnelObstacle fo(32.0f, domainLength);

  // pick up some points
  const float eps = domainLength / 64;
  const float my0 = -16.0;
  //const float yPlaneUp = 4.0;
  assertTrue(fo.sample(0, my0).second < eps);
  assertTrue(fo.isInside(0, my0 + 1) == true);
  assertTrue(fo.isInside(0, my0 - 1) == false);

  // check on the points from the border
  size_t szForEvery = 20;
  float h = domainLength / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * h - domainLength/2;
    float y = 0.0;
    if (x > -3.9 && x <= 3.9)
      assertTrue(fo.isInside(x, y));
    if (x < -4.1 || x >= 4.1)
      assertTrue(!fo.isInside(x, y));
  }

  for (size_t iy = 0; iy < szForEvery; ++iy) {
    float y = iy * h - domainLength/2;
    float x = 0.0;
    if (y > -15.9 && y <= 15.9)
      assertTrue(fo.isInside(x, y));
    if (y < -16.1 || y >= 16.1)
      assertTrue(!fo.isInside(x, y));
  }
  printOk();
}

void checkSample()
{
  /*
  // check on the points from the border
  size_t szForEvery = 20;
  size_t h = 2.0 * sqrtf(yPlaneUp - my0) / szForEvery;
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
  checkFind1();
  checkFind2();
}
