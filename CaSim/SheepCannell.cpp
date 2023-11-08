#include "SheepCannell.h"
#include <cmath>

SheepCannell::SheepCannell(std::map<std::string, double>& params) {
	kopen0 = params["kopen0"];
	ncaopen = params["ncaopen"];
	kopenmax = params["kopenmax"];
	kclose0 = params["kclose0"];
	ncaclose = params["ncaclose"];
}

double SheepCannell::_KClose(const std::unordered_map<std::string, double>& ions) {
	return kclose0 * std::pow(ions.at("Calcium") * 1e-3, ncaclose) * 1e-3;
}

double SheepCannell::_KOpen(const std::unordered_map<std::string, double>& ions) {
	return std::min(kopen0 * std::pow(ions.at("Calcium") * 1e-3, ncaopen), kopenmax) * 1e-3;
}