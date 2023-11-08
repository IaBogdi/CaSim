#pragma once

#include <nlohmann/json.hpp>
#include "CytosolSeries.h"
#include <vector>
#include <map>
#include <string>

class DyadSeries {
protected:
	int n_channels;
	bool is_cytosol_cpu;
	std::vector<uint64_t> dyad_dims;
	std::vector<uint64_t> channels_ions_dims;
	std::vector<uint64_t> channels_dims;
public:
	virtual int GetNumSRChannels() = 0;
	virtual bool UsesGPU() = 0;
	virtual void Reset() = 0;
	virtual void InitOpening(int, int) = 0;
	virtual double GetElementVolume() = 0;
	virtual std::vector<std::string> GetListofIons() = 0;
	virtual void GetEffluxes(double*&) = 0;
	virtual std::vector<double> GetTotalSRCurrent() = 0;
	virtual void RunMC(double,int) = 0;
	virtual void Update(double*&,double*&,const std::vector<double>&,const std::vector<double>&,double&) = 0;
	virtual double GetL() = 0;
	virtual void RunRD(double, int) = 0;
	virtual std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&) = 0;
	virtual std::map <std::string, std::vector<int> > GetChannelsStates(std::vector<std::string>&) = 0;
	virtual std::map <std::string, std::vector<double> > GetIonsNearChannels(std::vector<std::string>&) = 0;
	std::vector<uint64_t> GetDimensions() {
		return dyad_dims;
	}
	std::vector<uint64_t> GetChannelsDimensions() {
		return channels_dims;
	}
	std::vector<uint64_t> GetIonsNearChannelsDimensions() {
		return channels_ions_dims;
	}
	void IsCytosolGPU(bool ans) {
		is_cytosol_cpu = !ans;
	}
};

