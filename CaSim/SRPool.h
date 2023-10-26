#pragma once
#include "SRSeries.h"

#include <vector>
#include <string>
class SRPool :public SRSeries {
	std::vector<double> K_CQ;
	double V_dyad_element;
	double CQ_conc;
	std::vector<int> nCQ_sites;
	double V;
	std::vector<double> tau_refill;
	int nthreads;
	int nchannels;
	int nions;
	std::vector<double> ions;
	std::vector<double> ions_NSR;
	std::vector<double> IjSR_dyad;
	std::vector<double> ions_new;
	std::vector<double> ions_0;
	std::vector<std::string> ion_names;
	long _GetIdx(int, int, int);
	std::vector<uint64_t> sr_dims;
	bool is_const;
public:
	SRPool(nlohmann::json&, int,double,int, int);
	void Run(double, int);
	void Reset();
	bool UsesGPU();
	std::vector<double>& GetIons();
	void Update(const std::vector<double>&, const std::vector<double>&);
	std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&);
	std::vector<uint64_t> GetDimensions();
};

