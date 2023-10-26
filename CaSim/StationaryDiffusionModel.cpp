#include "StationaryDiffusionModel.h"

std::vector<double> StationaryDiffusionModel::GetConcentrations(std::vector < std::shared_ptr < KineticModel > >& channels) {
	std::vector<double> out(channels.size(), 0);
	for (int j = 0; j < out.size(); ++j) {
		for (int i = 0; i < channels.size(); ++i)
			out[j] += channels[i]->isOpen() * added_concentrations[i][j];
		out[j] += background;
	}
	return out;
}