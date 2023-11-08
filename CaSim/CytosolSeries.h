#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <memory>

#include "DyadSeries.h"
#include "SRSeries.h"

class CytosolSeries
{
public:
	virtual bool UsesGPU() = 0;
	virtual void Reset() = 0;
	virtual void GetIonsBuffersandV(double*&, double*&,double&) = 0;
	virtual std::vector<double>& GetExtraCellularIons() = 0;
	virtual void RunRD(double, int) = 0;
	virtual void Update(double*&,const std::vector<double>&) = 0;
	virtual std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&) = 0;
	virtual std::vector<uint64_t> GetDimensions() = 0;
};

