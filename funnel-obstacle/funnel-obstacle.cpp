/*
 *  funnel-obstacle.cpp
 *  Part of CTC/funnel-obstacle/
 *
 *  Created and authored by Kirill Lykov on 2014-07-31.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
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

Grid::Grid(size_t szY, size_t szX)
: m_data(szY * szX), m_n1(szY), m_n2(szX)
{
}

const float& Grid::operator() (size_t i, size_t j) const
{
  assert(i < m_n1 && j < m_n2);
  return m_data[i + m_n1 * j];
}

float& Grid::operator() (size_t i, size_t j)
{
  assert(i < m_n1 && j < m_n2);
  return m_data[i + m_n1 * j];
}

size_t Grid::size(size_t index) const
{
  assert(index <= 2);
  return index == 0 ? m_n1 : m_n2;
}

FunnelObstacle::FunnelObstacle(const float plength, const float domainLengthX, const float domainLengthY,
                               const size_t gridResolutionX, const size_t gridResolutionY)
:  m_grid(gridResolutionX, gridResolutionY), m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0),
   m_domainLength{domainLengthX, domainLengthY},
   m_skinWidth{0.0f, 0.0f}, m_obstacleDetalization(2000)
{
  m_skinWidth[0] = fabs(y2x(m_yPlaneUp, m_y0) - m_domainLength[0]/2.0f);
  m_skinWidth[1] = (m_domainLength[1] - plength) / 2;
  m_yPlaneUp = m_domainLength[1]/2.0f - m_skinWidth[1];
  m_y0 = fabs(-m_domainLength[1]/2.0f + m_skinWidth[1]);

  float h[] = {m_domainLength[0] / m_grid.size(1), m_domainLength[1] / m_grid.size(0)};
  initInterface();
  for (size_t iy = 0; iy < m_grid.size(0); ++iy) {
    for (size_t ix = 0; ix <  m_grid.size(1); ++ix) {
      float point[] = {ix * h[0]- m_domainLength[0]/2.0f, iy * h[1] - m_domainLength[1]/2.0f};
      float dist = calcDist(point[0], point[1]);
      assert(!std::isnan(dist));
      m_grid(iy, ix) = dist;
    }
  }
}

FunnelObstacle::FunnelObstacle(const float plength, const float domainLengthX,
                               const float domainLengthY, const size_t gridResolutionX,
                               const size_t gridResolutionY, const std::string& fileName)
:  m_grid(gridResolutionX, gridResolutionY), m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0),
   m_domainLength{domainLengthX, domainLengthY},
   m_obstacleDetalization(2000)
{
  m_skinWidth[0] = fabs(y2x(m_yPlaneUp, m_y0) - m_domainLength[0]/2.0f);
  m_skinWidth[1] = (m_domainLength[1] - plength) / 2;
  m_yPlaneUp = m_domainLength[1]/2.0f - m_skinWidth[1];
  m_y0 = fabs(-m_domainLength[1]/2.0f + m_skinWidth[1]);

  read(fileName);
}

bool FunnelObstacle::isInside(const float x, const float y) const
{
  if (insideBoundingBox(x, y))
    return sample(x, y).first;
  return false;
}

std::pair<bool, float> FunnelObstacle::sample(const float x, const float y) const
{
  assert(insideBoundingBox(x, y));
  float hy = m_domainLength[1] / (m_grid.size(0) - 1.0);
  float hx = m_domainLength[0] / (m_grid.size(1) - 1.0);

  // shift origin to the left bottom of the BB
  float xShift = x + m_domainLength[0]/2;
  float yShift = y + m_domainLength[1]/2;
  assert(xShift >= 0.0 && xShift >= 0.0);

  size_t ix, iy;
  // index_x = floor( p_x / h_x )
  ix = static_cast<int>(xShift / hx);
  iy = static_cast<int>(yShift / hy);

  float d =  bilinearInterpolation(xShift, yShift, hx, hy, ix, iy);

  /*float res = std::numeric_limits<float>::infinity();
  size_t iminX, iminY = 0;
  for (size_t i = 0; i < 4; ++i) {
    size_t ishX = i / 2, ishY = i % 2;
    std::cout << ishX << " " << ishY << std::endl;
    float d = dist(xShift, yShift, (ix + ishX)*hx, (iy + ishY)*hy);
    if (res > d) {
      res = d;
      iminX = ix + ishX;
      iminY = iy + ishY;
    }
  }

  assert(iminX < m_grid.size(0) && iminY < m_grid.size(1));
  float dist2 = m_grid(iminY, iminX);*/

  return std::pair<bool, float>(d < 0.0, fabs(d));
}

bool FunnelObstacle::isBetweenLayers(const float x, const float y,
                                     const float bottom, const float up) const
{
    assert(bottom < up && up < std::min(m_skinWidth[0], m_skinWidth[1]));
    std::pair<bool, float> res = sample(x, y);
    if (!res.first) //outside
        return res.second >= bottom && res.second <= up;
    return false;
}

void FunnelObstacle::initInterface()
{
  size_t szForEvery = m_obstacleDetalization / 2;
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

  assert(m_interface.size() == m_obstacleDetalization);
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

float FunnelObstacle::bilinearInterpolation(float x, float y, float hx, float hy, size_t ix, size_t iy) const
{
  float x1 = hx * ix, x2 = hx * (ix + 1);
  float y1 = hy * iy, y2 = hy * (iy + 1);

  float fr1 = m_grid(iy,ix) * (y2 - y) / (y2 - y1) + m_grid(iy + 1, ix) * (y - y1) / (y2 - y1);
  float fr2 = m_grid(iy, ix + 1) * (y2 - y) / (y2 - y1) + m_grid(iy + 1, ix + 1) * (y - y1) / (y2 - y1);

  float fp = fr1 * (x2 - x) /(x2 - x1) + fr2 * (x - x1) / (x2 - x1);
  return fp;
}

bool FunnelObstacle::insideBoundingBox(const float x, const float y) const
{
  if (x >= -m_domainLength[0]/2.0 && x <= m_domainLength[0]/2.0
   && y >= -m_domainLength[1]/2.0 && y <= m_domainLength[1]/2.0)
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

  for (size_t iy = 0; iy < m_grid.size(0); ++iy) {
    for (size_t ix = 0; ix < m_grid.size(1); ++ix) {
      fprintf(f, "   %e", m_grid(iy,ix));
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

  for (size_t iy = 0; iy < m_grid.size(0); ++iy) {
    for (size_t ix = 0; ix < m_grid.size(1); ++ix) {
      float val;
      int result  = fscanf(f, "   %e", &val);
      assert(result == 1);
      m_grid(iy, ix) = val;
    }

  }
  fclose(f);
}

bool FunnelObstacle::operator== (const FunnelObstacle& another)
{
  if (m_yPlaneUp != another.m_yPlaneUp || m_yPlaneDown != another.m_yPlaneDown || m_y0 != another.m_y0 ||
      m_domainLength[0] != another.m_domainLength[0] || m_skinWidth[2] != another.m_skinWidth[2] ||
      m_obstacleDetalization != another.m_obstacleDetalization)
    return false;

  for (size_t iy = 0; iy < m_grid.size(0); ++iy)
    for (size_t ix = 0; ix < m_grid.size(1); ++ix)
      if (fabs(m_grid(iy, ix) - another.m_grid(iy, ix)) > 1e-4)
        return false;
  return true;
}

// **************** RowFunnelObstacle *******************

RowFunnelObstacle::RowFunnelObstacle(const float plength, const float domainLengthX, const float domainLengthY,
                                     const size_t gridResolutionX, const size_t gridResolutionY)
: m_funnelObstacle(plength, domainLengthX, domainLengthY, gridResolutionX, gridResolutionY)
{}

float RowFunnelObstacle::getOffset(float x) const
{
    float h = x > 0.0f ? 0.5f : -0.5f;
    return -trunc(x / m_funnelObstacle.getDomainLength(0) + h) * m_funnelObstacle.getDomainLength(0);
}

bool RowFunnelObstacle::isBetweenLayers(const float x, const float y,
                                     const float bottom, const float up) const
{
    if (!insideBoundingBox(x, y))
        return false;

    float xShifted = x + getOffset(x);
    return m_funnelObstacle.isBetweenLayers(xShifted, y, bottom, up);
}

int RowFunnelObstacle::getBoundingBoxIndex(const float x, const float y) const
{
    // the row center is at (0.0, 0.0), thus first check that y is in the box
    // than find index
    if (!m_funnelObstacle.insideBoundingBox(0.0f, y))
        return std::numeric_limits<int>::max();

    float h = x > 0.0f ? 0.5f : -0.5f;
    int res =  static_cast<int>(x / m_funnelObstacle.getDomainLength(0) + h);
    return res;
}

bool RowFunnelObstacle::isInside(const float x, const float y) const
{
    return sample(x, y).first;
}

std::pair<bool, float> RowFunnelObstacle::sample(const float x, const float y) const
{
    if (!insideBoundingBox(x, y))
        return std::pair<bool, float>(false, std::numeric_limits<float>::infinity());

    float xShifted = x + getOffset(x);
    return m_funnelObstacle.sample(xShifted, y);
}

