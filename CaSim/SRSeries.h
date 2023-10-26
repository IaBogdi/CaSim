#pragma once
#include <nlohmann/json.hpp>
#include <vector>
#include "NSRSeries.h"
#include "DyadSeries.h"

class SRSeries {
public:
	virtual void Run(double,int) = 0;
	virtual void Reset() = 0;
	virtual bool UsesGPU() = 0;
	virtual void Update(const std::vector<double>&,const std::vector<double>&) = 0;
	virtual std::vector<double>& GetIons() = 0;
	virtual std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&) = 0;
	virtual std::vector<uint64_t> GetDimensions() = 0;
};

