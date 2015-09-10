/*
 *  obst-check.cpp
 *  Part of uDeviceX/funnel-obstacle/
 *
 *  Created and authored by Kirill Lykov on 2014-07-30.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <limits>
#include <vector>
#include <cmath>
#include "funnel-obstacle.h"

#define printOk() do{std::cout << "Quick test " << __func__ << ": OK\n";}while(0)

#define assertTrue(res) do{ if (!(res)) {std::cout << "Quick test " << __func__ << ": FAIL\n";} assert(res);}while(0)

void checkFunnelObstacleReadWrite()
{
  std::string inputFileName = "bla.dat";
  FunnelObstacle fo(16.0f, 20.0f, 40.0f, 128, 128);
  fo.write(inputFileName);

  FunnelObstacle fIn(16.0f, 20.0f, 40.0f, 128, 128, inputFileName);
  assertTrue(fIn == fo);
  printOk();
}

void checkFunnelObstacleFind1()
{
  FunnelObstacle fo(8.0f, 10.0f, 10.0f);

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

    float x = 0.0;
    if (y > -3.9 && y <= 3.9)
      assertTrue(fo.isInside(x, y));
    if (y < -4.1 || y >= 4.1)
      assertTrue(!fo.isInside(x, y));
  }
  printOk();
}

void checkFunnelObstaclNotSquare()
{
  float domainLenght[] = {5.0f, 10.0f};
  FunnelObstacle fo(8.0f, domainLenght[0], domainLenght[1], 32, 64);

  // pick up some points
  const float eps = 0.1;
  const float my0 = -4.0;
  //const float yPlaneUp = 4.0;
  assertTrue(fo.sample(0, my0).second < eps);
  assertTrue(fo.isInside(0, my0 + 1) == true);
  assertTrue(fo.isInside(0, my0 - 1) == false);

  // check on the points from the border
  size_t szForEvery = 20;
  float hx = domainLenght[0] / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * hx - domainLenght[0]/2.0;
    float y = 0.0;
    if (x > -1.9 && x <= 1.9)
      assertTrue(fo.isInside(x, y));
    if (x < -2.1 || x >= 2.1)
      assertTrue(!fo.isInside(x, y));
  }

  float hy = domainLenght[1] / szForEvery;
  for (size_t iy = 0; iy < szForEvery; ++iy) {
    float y = iy * hy - domainLenght[1]/2.0f;
    float x = 0.0;
    if (y > -3.9 && y <= 3.9)
      assertTrue(fo.isInside(x, y));
    if (y < -4.1 || y >= 4.1)
      assertTrue(!fo.isInside(x, y));
  }
  printOk();
}

template <class Obstacle>
void checkObstacle(Obstacle& fo, float xc, float yc)
{
  float domainLength = 39.9f;

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
    float x = ix * h - domainLength/2 + xc;
    float y = yc;
    if (x > -3.9 + xc && x <= 3.9 + xc)
      assertTrue(fo.isInside(x, y));
    if (x < -4.1 + xc || x >= 4.1 + xc)
      assertTrue(!fo.isInside(x, y));
  }

  for (size_t iy = 0; iy < szForEvery; ++iy) {
    float y = iy * h - domainLength/2 + yc;
    float x = xc;
    if (y > -15.9 + yc && y <= 15.9 + yc)
      assertTrue(fo.isInside(x, y));
    if (y < -16.1 + yc || y >= 16.1 + yc)
      assertTrue(!fo.isInside(x, y));
  }
}


void checkFunnelObstacleFind2()
{
  float domainLength = 40.0f;
  FunnelObstacle fo(32.0f, domainLength, domainLength);

  checkObstacle(fo, 0.0f, 0.0f);
  printOk();
}

void checkFunnelObstacleSample()
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

void checkRowFunnelObstacle1()
{
    // the behavior in simple case must be the same
    float domainLength = 40.0f;
    RowFunnelObstacle fo(32.0f, domainLength, domainLength);

    checkObstacle(fo, 0.0f, 0.0f);

    //shift by period, should give the same result
    checkObstacle(fo, domainLength, 0.0f);
    checkObstacle(fo, -domainLength, 0.0f);

    assertTrue(fo.getBoundingBoxIndex(0.0, 100.0f) == std::numeric_limits<int>::max());

    assertTrue(fo.getBoundingBoxIndex(0.0, 0.0) == 0);

    assertTrue(fo.getBoundingBoxIndex(25.0f, 0.0) == 1);
    assertTrue(fo.getBoundingBoxIndex(-25.0f, 0.0) == -1);
    printOk();
}

void checkRowFunnelObstacle2()
{
    // the behavior in simple case must be the same
    float domainLength = 20.0f;
    RowFunnelObstacle funnelLS(5.0f, 10.0f, 10.0f, 64, 64);

    assertTrue(funnelLS.isInside(0.0f, 0.0f));
    assertTrue(funnelLS.isInside(-domainLength/2.0f, 0.0f));
    assertTrue(funnelLS.isInside(domainLength/2.0f, 0.0f));

    for (int i = 0; i < 4; ++i) {
        float x = 0.0f;
        float h = 0.5;
        float y = -2.7f - i*h;
        assertTrue(funnelLS.isBetweenLayers(x, y, i*h , (i+1)*h));
    }

    assertTrue(!funnelLS.isBetweenLayers(0.0, -2.6, 1, 2));
    assertTrue(!funnelLS.isBetweenLayers(0.0, -5.5, 0, 1));

    //the same for shifted
    for (int i = 0; i < 4; ++i) {
        float x = -domainLength/2.0f + 1e-4;
        float h = 0.5;
        float y = -2.7f - i*h;
        assertTrue(funnelLS.isBetweenLayers(x, y, i*h , (i+1)*h));
    }

    printOk();
}

int main()
{
  checkFunnelObstacleReadWrite();
  checkFunnelObstacleFind1();
  checkFunnelObstaclNotSquare();
  checkFunnelObstacleFind2();
  checkRowFunnelObstacle1();
  checkRowFunnelObstacle2();
}
