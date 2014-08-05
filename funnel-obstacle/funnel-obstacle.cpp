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

FunnelObstacle::FunnelObstacle(const float plength, const float domainLength, const size_t gridResolution)
:  m_grid(gridResolution, gridResolution), m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0), m_domainLength(domainLength),
   m_obstacleDetalization(2000)
{
  float hy = (m_domainLength - plength) / 2;
  m_yPlaneUp = m_domainLength/2.0f - hy;
  m_y0 = fabs(-m_domainLength/2.0f + hy);

  float h = m_domainLength / m_grid.size(0);
  initInterface();
  for (size_t iy = 0; iy < m_grid.size(0); ++iy) {
    for (size_t ix = 0; ix <  m_grid.size(1); ++ix) {
      float point[] = {ix * h - m_domainLength/2.0f, iy * h - m_domainLength/2.0f};
      float dist = calcDist(point[0], point[1]);
      assert(!std::isnan(dist));
      m_grid(iy, ix) = dist;
    }
  }
}

FunnelObstacle::FunnelObstacle(const float plength, const float domainLength, const size_t gridResolution, const std::string& fileName)
:  m_grid(gridResolution, gridResolution), m_yPlaneUp(0), m_yPlaneDown(0.0), m_y0(0.0), m_domainLength(domainLength),
   m_obstacleDetalization(2000)
{
  float hy = (m_domainLength - plength) / 2;
  m_yPlaneUp = m_domainLength/2 - hy;
  m_y0 = fabs(-m_domainLength/2 + hy);

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
  assert(insideBB(x, y));
  float hy = m_domainLength / (m_grid.size(0) - 1.0);
  float hx = m_domainLength / (m_grid.size(1) - 1.0);

  // shift origin to the left bottom of the BB
  float xShift = x + m_domainLength/2;
  float yShift = y + m_domainLength/2;
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

bool FunnelObstacle::insideBB(const float x, const float y) const
{
  float half = m_domainLength / 2.0;
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
  if (m_yPlaneUp != another.m_yPlaneUp || m_yPlaneDown != another.m_yPlaneDown || m_y0 != another.m_y0)
    return false;

  for (size_t iy = 0; iy < m_grid.size(0); ++iy)
    for (size_t ix = 0; ix < m_grid.size(1); ++ix)
      if (fabs(m_grid(iy, ix) - another.m_grid(iy, ix)) > 1e-4)
        return false;
  return true;
}

// **************** RowFunnelObstacle *******************

RowFunnelObstacle::RowFunnelObstacle(const float plength, const float domainLength, const size_t gridResolution)
: m_funnelObstacle(plength, domainLength, gridResolution)
{}

float RowFunnelObstacle::getOffset(float x) const
{
    float h = x > 0.0f ? 0.5f : -0.5f;
    return -trunc(x / m_funnelObstacle.getDomainLength(0) + h) * m_funnelObstacle.getDomainLength(0);
}

bool RowFunnelObstacle::isInside(const float x, const float y) const
{
    float xShifted = x + getOffset(x);
    return m_funnelObstacle.isInside(xShifted, y);
}

std::pair<bool, float> RowFunnelObstacle::sample(const float x, const float y) const
{
    float xShifted = x + getOffset(x);
    return m_funnelObstacle.sample(xShifted, y);
}

