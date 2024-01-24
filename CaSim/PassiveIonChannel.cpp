#include "PassiveIonChannel.h"
#include "TwoStateMWC.h"
#include "SingleSiteMWC.h"
#include "SingleSiteMWCwCaL.h"
#include "SheepCannell.h"
#include "TwoStateSplineModel.h"
#include "TwoStateaHTG.h"
#include <map>

PassiveIonChannel::PassiveIonChannel(std::string model_name, std::vector<double>& coords, std::map<std::string, double>& params, std::vector<double>& G,
	std::string& sopen, std::string& sclose) {
	g = G;
	if (model_name == "TwoStateMWC")
		model = std::make_unique<TwoStateMWC>(params);
	if (model_name == "SingleSiteMWC")
		model = std::make_unique<SingleSiteMWC>(params);
	if (model_name == "SingleSiteMWCCaL")
		model = std::make_unique<SingleSiteMWCwCaL>(params);
	if (model_name == "SheepCannell")
		model = std::make_unique<SheepCannell>(params);
	if (model_name == "TwoStateSpline")
		model = std::make_unique<TwoStateSplineModel>(sopen, sclose);
	if (model_name == "TwoStateaHTG")
		model = std::make_unique<TwoStateaHTG>(params);
	coordinates = std::make_unique<Point>(coords);
}

double PassiveIonChannel::Conductance(double jsr, double ds, int idx) {
	return g[idx] * (jsr - ds);
}

double PassiveIonChannel::Flux(double jsr, double ds,int idx) {
	return model->GetMacroState()*Conductance(jsr,ds,idx);
}