/*
 * funnel-obstacle.cpp
 *
 *  Created on: Jul 31, 2014
 *      Author: kirill
 */
#include "funnel-obstacle.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cassert>

#include <algorithm>
#include <vector>
#include <limits>

typedef std::vector< std::pair<float, float> >::const_iterator RIterator;

namespace {
  // functions parameters are hidden
  float boxLength = 10.0;
  size_t gridSize = 64;
  size_t obstacleDetalization = 2000;

  float leftBottom[] = {-boxLength/2, -boxLength/2};

  float x2y(float x, float y0)
  {
    return x * x - y0;
  }

  float y2x(float y, float y0)
  {
    return sqrtf(y + y0);
  }

  std::pair<float, float> mkParab(float x, float y0)
  {
    return std::pair<float, float>(x, x2y(x, y0));
  }

  std::pair<float, float> mkUpLine(float x, float yPlaneUp)
  {
    return std::pair<float, float>(x, yPlaneUp);
  }

  float dist(float x1, float y1, float x2, float y2)
  {
    return sqrtf( (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
  }
}

FunnelObstacle::FunnelObstacle()
:  m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0)
{
}

FunnelObstacle::FunnelObstacle(const float plength)
:  m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0)
{
  float hy = (boxLength - plength) / 2;
  m_yPlaneUp = boxLength/2 - hy;
  m_y0 = fabs(-boxLength/2 + hy);

  float h = boxLength / gridSize;
  initInterface();
  for (size_t iy = 0; iy < gridSize; ++iy) {
    for (size_t ix = 0; ix < gridSize; ++ix) {
      float point[] = {ix * h + leftBottom[0], iy * h + leftBottom[1]};
      float dist = calcDist(point[0], point[1]);
      assert(!std::isnan(dist));
      m_grid.data[iy][ix] = dist;
    }
  }
}

FunnelObstacle::FunnelObstacle(const float plength, const std::string& fileName)
:  m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0)
{
  float hy = (boxLength - plength) / 2;
  m_yPlaneUp = boxLength/2 - hy;
  m_y0 = fabs(-boxLength/2 + hy);

  read(fileName);
}

bool FunnelObstacle::isInside(const float x, const float y) const
{
  if (insideBB(x, y))
    return sample(x, y).first;
  return false;
}

std::pair<bool, float> FunnelObstacle::sample(const float x, const float y) const
{
  float h = boxLength / (gridSize - 1.0);

  // shift origin to the left bottom of the BB
  float xShift = x + boxLength/2;
  float yShift = y + boxLength/2;
  assert(xShift >= 0.0 && xShift >= 0.0);

  size_t ix, iy;
  // index_x = floor( p_x / h_x )
  ix = static_cast<int>(xShift / h);
  iy = static_cast<int>(yShift / h);

  float res = std::numeric_limits<float>::infinity();
  size_t iminX, iminY = 0;
  for (size_t i = 0; i < 4; ++i) {
    size_t ishX = i / 2, ishY = i % 2;
    float d = dist(xShift, yShift, (ix + ishX)*h, (iy + ishY)*h);
    if (res > d) {
      res = d;
      iminX = ix + ishX;
      iminY = iy + ishY;
    }
  }

  assert(iminX < gridSize && iminY < gridSize);
  float dist = m_grid.data[iminY][iminX];

  return std::pair<bool, float>(dist < 0.0, fabs(dist));
}

void FunnelObstacle::initInterface()
{
  size_t szForEvery = obstacleDetalization / 2;
  float h = 2.0 * y2x(m_yPlaneUp, m_y0) / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * h - y2x(m_yPlaneUp, m_y0);
    m_interface.push_back(mkParab(x, m_y0));
  }

  h = 2.0 * y2x(m_yPlaneUp, m_y0) / szForEvery;
  for (size_t ix = 0; ix < szForEvery; ++ix) {
    float x = ix * h - y2x(m_yPlaneUp, m_y0);
    m_interface.push_back(mkUpLine(x, m_y0));
  }

  assert(m_interface.size() == obstacleDetalization);
}

float FunnelObstacle::calcDist(const float x, const float y) const
{
  float minDist = std::numeric_limits<float>::infinity();
  for (RIterator it = m_interface.begin(); it != m_interface.end(); ++it) {
    float d = sqrt((x - it->first) * (x - it->first) + (y - it->second) * (y - it->second));
    minDist = std::min(minDist, d);
  }
  if ((y > x2y(x, m_y0)) && (y < m_yPlaneUp))
    minDist *= -1.0;

  return minDist;
}

bool FunnelObstacle::insideBB(const float x, const float y) const
{
  float half = boxLength / 2.0;
  if (x >= -half && x <= half
   && y >= -half && y <= half)
    return true;
  return false;
}

void FunnelObstacle::write(const std::string& fileName) const
{
  FILE * f = fopen(fileName.c_str(), "w");
  if (f == NULL)
  {
    printf("I could not open the file <%s>\n", fileName.c_str());
    printf("Aborting now.\n");
    abort();
  }

  for (size_t iy = 0; iy < gridSize; ++iy) {
    for (size_t ix = 0; ix < gridSize; ++ix) {
      fprintf(f, "   %e", m_grid.data[iy][ix]);
    }
    fprintf(f, "\n");
  }
  fclose(f);
}

void FunnelObstacle::read(const std::string& fileName)
{
  FILE * f = fopen(fileName.c_str(), "r");
  if (f == NULL)
  {
    printf("I could not open the file <%s>\n", fileName.c_str());
    printf("Aborting now.\n");
    abort();
  }

  for (size_t iy = 0; iy < gridSize; ++iy) {
    for (size_t ix = 0; ix < gridSize; ++ix) {
      float val;
      int result  = fscanf(f, "   %e", &val);
      assert(result == 1);
      m_grid.data[iy][ix] = val;
    }

  }
  fclose(f);
}

bool FunnelObstacle::operator== (const FunnelObstacle& another)
{
  if (m_yPlaneUp != another.m_yPlaneUp || m_yPlaneDown != another.m_yPlaneDown || m_y0 != another.m_y0)
    return false;

  for (size_t iy = 0; iy < gridSize; ++iy)
    for (size_t ix = 0; ix < gridSize; ++ix)
      if (fabs(m_grid.data[iy][ix] - another.m_grid.data[iy][ix]) > 1e-4)
        return false;
  return true;
}
