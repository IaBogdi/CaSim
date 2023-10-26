#pragma once
#include "KineticModel.h"

#include <vector>
#include <cmath>
#include <string>
#include <memory>

class StationaryDiffusionModel {
protected:
	const double Far = 96485.33e-6;
	const double Pi = std::acos(-1);
	std::string channel_name;
	double background;
	std::vector< std::vector<double> > added_concentrations;
public:
	std::vector<double> GetConcentrations(std::vector < std::shared_ptr < KineticModel > >&);
};

