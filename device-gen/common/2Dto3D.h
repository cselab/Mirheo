#pragma once
#include <vector>
#include <string>

void conver2Dto3D(const int NX, const int NY, const float xextent, const float yextent, const std::vector<float>& slice,
                  const int NZ, const float zextent, const float zmargin, const std::string& fileName);
