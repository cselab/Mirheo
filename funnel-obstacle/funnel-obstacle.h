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
#include <limits>
#include <cassert>

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

  float m_yPlaneUp, m_yPlaneDown, m_y0, m_domainLength[2], m_skinWidth[2];
  const size_t m_obstacleDetalization;

  std::vector< std::pair <float, float> > m_interface;
  void initInterface();

  float calcDist(const float x, const float y) const;
  float bilinearInterpolation(float x, float y, float hx, float hy, size_t i, size_t j) const;

  void read(const std::string& fileName);

  FunnelObstacle(const FunnelObstacle&);
  FunnelObstacle& operator= (const FunnelObstacle&);

public:

  FunnelObstacle(const float plength, const float domainLengthX,
                 const float domainLengthY, const size_t gridResolutionX = 32, const size_t gridResolutionY = 64);
  FunnelObstacle(const float plength, const float domainLengthX,
                 const float domainLengthY, const size_t gridResolutionX, const size_t gridResolutionY,
                 const std::string& fileName);

  bool isInside(const float x, const float y) const;
  std::pair<bool, float> sample(const float x, const float y) const;

  bool insideBoundingBox(const float x, const float y) const;

  float getDomainLength(size_t direct) const
  {
      assert(direct < 3);
      return m_domainLength[direct];
  }

  /**
   * the min distance from the interface to the border of bounding box
   */
  void getSkinWidth(float& x, float& y) const
  {
      x = m_skinWidth[0]; y = m_skinWidth[1];
  }

  void write(const std::string& fileName) const;

  bool operator== (const FunnelObstacle& another);
};

/**
 * Row of funnel obstacles in the X direction
 */
class RowFunnelObstacle
{
    size_t m_nBlocks;
    FunnelObstacle m_funnelObstacle;
public:
    RowFunnelObstacle(const float plength, const float domainLengthX, const float domainLengthY,
            size_t gridResolutionX = 32, const size_t gridResolutionY = 64);

    /**
     * return offset to shift the point into the bounding box for obstacle with index 0
     */
    float getOffset(float x) const;

    void getSkinWidth(float& x, float& y) const
    {
        m_funnelObstacle.getSkinWidth(x, y);
    }

    float getCoreDomainLength(size_t direct) const
    {
        return m_funnelObstacle.getDomainLength(direct);
    }

    bool insideBoundingBox(const float x, const float y) const
    {
        return (abs(getBoundingBoxIndex(x, y)) != std::numeric_limits<int>::max());
    }

    /**
     * return bbIndex which is [-I, I] if point belongs to one
     * of the bounding boxes for the row of obstacles, otherwise return inf
     */
    int getBoundingBoxIndex(const float x, const float y) const;

    bool isInside(const float x, const float y) const;
    std::pair<bool, float> sample(const float x, const float y) const;
};

#endif /* LS_OBSTACLE_H_ */
