#pragma once
#include "CalciumReleaseSite.h"
#include "StationaryDiffusionModel.h"
#include "KineticModel.h"
#include "LinearNaraghiNeher.h"
#include <memory>

class QuasyStationaryReleaseSite :public CalciumReleaseSite {
	std::unique_ptr < StationaryDiffusionModel > diffusion;
	std::vector < std::shared_ptr < KineticModel > >   channels;
	std::string channel_name;
	std::vector<std::vector<double> > ions;
	double max_time;
	int num_channels = 0;
	std::vector<double> GetOutVector(double, std::vector<double>&);
	std::vector<double> GetOutVector(double);
public:
	void RunSimulation(datavector& channels, datavector& concentrations_dc, datavector& concentrations_lumen, datavector& concentrations_cyt, int, ParallelRng*);
	QuasyStationaryReleaseSite(nlohmann::json&);
	std::vector<std::vector<int> > GetChannelsMacroStatesMap(std::string);
	int GetNChannels();
};