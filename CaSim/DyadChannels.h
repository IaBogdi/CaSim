#pragma once
#include "IonChannel.h"
#include "ParallelRng.h"
#include <nlohmann/json.hpp>
#include <memory>

class DyadRD;

class DyadChannels
{
	std::vector < std::vector < std::shared_ptr < IonChannel > > > SRChannels;
	std::vector < std::vector < std::shared_ptr < IonChannel > > > SLChannels;
	std::vector < std::vector <double> > MCIntegral;
	std::unique_ptr<ParallelRng> rng;
	std::shared_ptr <IonChannel> GetIonChannel(std::vector < std::vector < std::shared_ptr < IonChannel > > >&, int, int);
	double Cab;
	double Mg0;
	double CaM0;
	double Cajsrb;
	int n_threads;
	int n_ions;
	bool is_step;
	int n_ions_lum;
	std::vector<std::vector<double> > ions_temp;
	void _Initialize(nlohmann::json&, int);
public:
	DyadChannels(nlohmann::json&, nlohmann::json&, int);
	DyadChannels(nlohmann::json&, nlohmann::json&, int, std::vector<float>&);
	DyadChannels(nlohmann::json&, int);
	std::shared_ptr <IonChannel> GetSRIonChannel(int, int);
	std::shared_ptr <IonChannel> GetSLIonChannel(int, int);
	int GetNSRIonChannels();
	int GetNSLIonChannels();
	void InitOpening(int, int);
	void RunMC(double, int,std::vector<double>&);
	void RunMC(double, int, std::vector<double>&, const std::vector<double>&);
};

