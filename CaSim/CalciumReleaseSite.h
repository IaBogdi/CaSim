#pragma once
#include <nlohmann/json.hpp>
#include <map>
#include <vector>
#include <string>
#include "ParallelRng.h"


typedef std::map < std::string, std::vector < std::vector < double > > > datavector;
typedef std::map < std::string, std::vector < std::vector < std::vector < double > > > > dataset;



class CalciumReleaseSite {
protected:
	double ca_b;
public:
	virtual void RunSimulation(datavector& channels, datavector& concentrations_dc, datavector& concentrations_lumen, datavector& concentrations_cyt, int, ParallelRng*) = 0;
	virtual int GetNChannels() = 0;
	virtual std::vector<std::vector<int> > GetChannelsMacroStatesMap(std::string) = 0;
};

