/*
 * ls-obstacle.h
 *
 *  Created on: Jul 31, 2014
 *      Author: kirill
 */

#ifndef LS_OBSTACLE_H_
#define LS_OBSTACLE_H_

#include <vector>
#include <utility>
#include <string>

class Grid
{
  std::vector<float> m_data; // YX
  const size_t m_n1, m_n2;

  Grid(const Grid&);
  Grid& operator= (const Grid&);
public:
  Grid(size_t szY, size_t szX);
  const float& operator() (size_t i, size_t j) const;
  float& operator() (size_t i, size_t j);
  size_t size(size_t index) const;
};

class FunnelObstacle
{
  Grid m_grid;

  float m_yPlaneUp, m_yPlaneDown, m_y0, m_domainLength;
  const size_t m_obstacleDetalization;

  std::vector< std::pair <float, float> > m_interface;
  void initInterface();

  float calcDist(const float x, const float y) const;
  float bilinearInterpolation(float x, float y, float hx, float hy, size_t i, size_t j) const;
  bool insideBB(const float x, const float y) const;

  void read(const std::string& fileName);

  FunnelObstacle(const FunnelObstacle&);
  FunnelObstacle& operator= (const FunnelObstacle&);

public:

  FunnelObstacle(const float plength, const float domainLength, const size_t gridResolution = 64);
  FunnelObstacle(const float plength, const float domainLength, const size_t gridResolution, const std::string& fileName);

  bool isInside(const float x, const float y) const;
  std::pair<bool, float> sample(const float x, const float y) const;

  float getDomainLength(size_t direct) const
  {
      return m_domainLength;
  }

  void write(const std::string& fileName) const;

  bool operator== (const FunnelObstacle& another);
};

/**
 * Row of funnel obstacles in the X direction
 */
class RowFunnelObstacle
{
    FunnelObstacle m_funnelObstacle;
public:
    RowFunnelObstacle(const float plength, const float domainLength, const size_t gridResolution = 64);

    float getOffset(float x) const;

    bool isInside(const float x, const float y) const;
    std::pair<bool, float> sample(const float x, const float y) const;
};

#endif /* LS_OBSTACLE_H_ */
