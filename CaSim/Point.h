#pragma once

#include <vector>

class Point {
	double x;
	double y;
	double z;
public:
	Point(double x2, double y2, double z2) : x(x2), y(y2), z(z2) {}
	Point(std::vector<double>&);
	double d(Point&);
	std::vector<int> GetCoordsOnGrid(double, double, double);
};

