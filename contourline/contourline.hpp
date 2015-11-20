#pragma once

#include <vector>
#include <utility>

typedef std::pair<float, float> Point;
typedef std::vector<Point> Line;
typedef std::vector<Line> Lines;

Lines contourlines(float* array, int n0, int n1, float h0, float h1, float value);
