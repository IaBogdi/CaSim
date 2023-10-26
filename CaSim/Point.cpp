#include "Point.h"
#include <cmath>


Point::Point(std::vector<double>& coords) {
    x = coords[0];
    y = coords[1];
    z = coords[2];
}

double Point::d(Point& p)
{
    double dd = (x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) + (z - p.z) * (z - p.z);
    return std::sqrt(dd);
}

std::vector<int> Point::GetCoordsOnGrid(double dx, double dy, double dz) {
    int nx = (x + 0.5 * dx) / dx,
        ny = (y + 0.5 * dy) / dy,
        nz = (z + 0.5 * dz) / dz;
    std::vector<int> ans{ nx, ny, nz };
    return ans;
}