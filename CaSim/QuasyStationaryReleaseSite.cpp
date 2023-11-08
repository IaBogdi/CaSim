#include "QuasyStationaryReleaseSite.h"
#include "Point.h"
#include "TwoStateMWC.h"
#include "SingleSiteMWC.h"

#include <fstream>
#include <iostream>

QuasyStationaryReleaseSite::QuasyStationaryReleaseSite(nlohmann::json& file) {
	std::map < std::string, std::vector<std::shared_ptr < Point > > > coordinates;
	double current,dca,ca_open;
	int calc_channels = 0;
	for (auto const& el : file["Calcium Release Site"]["Parameters"]["Channels"].items()) {
		++calc_channels;
		channel_name = el.key();
	}
	if (calc_channels != 1) {
		std::cout << "Wrong number of channels. Stationary diffusion models accept exactly one ion channel" << std::endl;
	}
	max_time = file["Calcium Release Site"]["Max Time"];
	std::string filename(file["Calcium Release Site"]["Parameters"]["Channels"][channel_name]["Coordinates"]);
	std::ifstream in(filename.c_str());
	std::vector<std::shared_ptr < Point > > points;
	double x, y;
	auto pars_channel = file["Calcium Release Site"]["Parameters"]["Channels"][channel_name]["Parameters"].get<std::map<std::string, double>>();
	while (in >> x >> y) {
		++num_channels;
		points.push_back(std::make_shared<Point>(x, y, 0));
		if (file["Calcium Release Site"]["Parameters"]["Channels"][channel_name]["Model"] == "TwoStateMWC")
			channels.push_back(std::make_shared<TwoStateMWC>(pars_channel));
		if (file["Calcium Release Site"]["Parameters"]["Channels"][channel_name]["Model"] == "SingleSiteMWC")
			channels.push_back(std::make_shared<SingleSiteMWC>(pars_channel));
	}
	in.close();
	//add other ions
	std::unordered_map<std::string, double> ions_t;
	for (auto const& el : file["Calcium Release Site"]["Parameters"]["Dyadic Cleft"]["Ions"].items())
			ions_t[el.key()] = el.value()["Concentration"];
	ions.resize(num_channels, ions_t);
	current = file["Calcium Release Site"]["Parameters"]["Channels"][channel_name]["Current"];
	ca_b = file["Calcium Release Site"]["Parameters"]["Dyadic Cleft"]["Ions"]["Calcium"]["Concentration"];
	dca = file["Calcium Release Site"]["Parameters"]["Dyadic Cleft"]["Ions"]["Calcium"]["D"];
	auto buffers_crs = file["Calcium Release Site"]["Parameters"]["Dyadic Cleft"]["Buffers"].get<std::map < std::string, std::map< std::string, double> >>();
	ca_open = file["Calcium Release Site"]["Parameters"]["Open Concentration"];
	if (file["Calcium Release Site"]["Name"] == "LinearNaraghiNeher")
		diffusion = std::make_unique<LinearNaraghiNeher>(buffers_crs,points,current,ca_b,dca,ca_open);
}

std::vector<std::vector<int>> QuasyStationaryReleaseSite::GetChannelsMacroStatesMap(std::string name)
{
	std::vector<std::vector<int>> ans;
	for (auto c : channels) {
		std::vector<int> s;
		for (int i = 0; i < c->GetNStates(); ++i) 
			s.push_back(c->GetMacroState(i));
		ans.push_back(s);
	}
	return ans;
}

int QuasyStationaryReleaseSite::GetNChannels()
{
	return num_channels;
}

std::vector<double> QuasyStationaryReleaseSite::GetOutVector(double t, std::vector<double>& vec)
{
	std::vector<double> out(vec.size()+1);
	out[0] = t;
	for (int i = 0; i < vec.size(); ++i)
		out[i + 1] = vec[i];
	return out;
}

std::vector<double> QuasyStationaryReleaseSite::GetOutVector(double t)
{
	std::vector<double> out(channels.size() + 1);
	out[0] = t;
	for (int i = 0; i < channels.size(); ++i)
		out[i + 1] = channels[i]->GetState();
	return out;
}



void QuasyStationaryReleaseSite::RunSimulation(datavector& channels_out, datavector& concentrations_dc, datavector& concentrations_lumen, datavector& concentrations_cyt,int channel, ParallelRng* rng) {
	//set channels to closed state
	for (int i = 0; i < num_channels; ++i)
		ions[i]["Calcium"] = ca_b;
	for (int i = 0; i < num_channels; ++i)
		channels[i]->SetMacroState(0,rng->runif(),ions[i]);
	//find initial open channel
	int idx_start = channel; //rng->runifint(0, channels.size() - 1);
	channels[idx_start]->SetMacroState(1,rng->runif(),ions[idx_start]);
	int num_open = 1;
	double t = 0;
	channels_out[channel_name].push_back(GetOutVector(t));
	while (num_open >= 1 && t < max_time) {
		//Gillespie
		auto C = diffusion->GetConcentrations(channels);
		concentrations_dc["Calcium"].push_back(GetOutVector(t,C));
		double total_rate = 0;
		for (int i = 0; i < C.size(); ++i) {
			ions[i]["Calcium"] = C[i];
			total_rate += channels[i]->GetRate(ions[i]);
		}
		double dt = rng->rexp(1 / total_rate);
		t += dt;
		//choose next channel
		double r = rng->runif();
		double p = 0;
		int idx = 0;
		for (int i = 0; i < channels.size(); ++i) {
			if (p <= r && r <= p + channels[i]->GetRate(ions[i]) / total_rate) {
				idx = i;
				break;
			}
			p += channels[i]->GetRate(ions[i]) / total_rate;
		}
		int was_open = channels[idx]->isOpen();
		//make transition
		channels[idx]->MakeTransition(rng->runif());
		//check if the channel changed its macrostate
		num_open += std::pow(-1, was_open) * (was_open != channels[idx]->isOpen());
		channels_out[channel_name].push_back(GetOutVector(t));
	}
	//last output in case 
	auto C = diffusion->GetConcentrations(channels);
	concentrations_dc["Calcium"].push_back(GetOutVector(t, C));
}

