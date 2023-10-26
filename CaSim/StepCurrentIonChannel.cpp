#include "StepCurrentIonChannel.h"
#include "StepCurrentModel.h"

StepCurrentIonChannel::StepCurrentIonChannel(std::vector<double>& coords, double T, std::vector<double>& G) {
	g = G;
	model = std::make_unique<StepCurrentModel>(T);
	coordinates = std::make_unique<Point>(coords);
}

double StepCurrentIonChannel::Flux(double jsr, double ds, int idx) {
	return model->GetMacroState() * g[idx];
}