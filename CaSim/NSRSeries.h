#pragma once

#include <vector>

class NSRSeries
{
public:
	virtual bool UsesGPU() = 0;
	virtual void Reset() = 0;
	virtual std::vector<double> GetIons() = 0;
	virtual void Update(double*&, const std::vector<double>&) = 0;
};

