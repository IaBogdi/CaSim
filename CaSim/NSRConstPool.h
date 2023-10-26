#pragma once
#include <nlohmann/json.hpp>
#include "NSRSeries.h"
#include <vector>

class NSRConstPool : public NSRSeries {
	std::vector<double> ions;
	std::vector<double> ions_out;
public:
	NSRConstPool(nlohmann::json&, int);
	bool UsesGPU();
	std::vector<double> GetIons();
	void Update(double*&, const std::vector<double>&);
	void Reset();
};

