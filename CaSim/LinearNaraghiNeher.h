#pragma once
#include "StationaryDiffusionModel.h"
#include "Point.h"
#include <vector>
#include <nlohmann/json.hpp>
#include <map>
#include <cmath>
#include <string>


class LinearNaraghiNeher : public StationaryDiffusionModel {
public:
	LinearNaraghiNeher(std::map < std::string, std::map< std::string, double> >&, std::vector<std::shared_ptr < Point > > &, double,double,double,double);
};

