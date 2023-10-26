#include "TwoStateSplineModel.h"
#include <fstream>

TwoStateSplineModel::TwoStateSplineModel(std::string& s1, std::string &s2) {
	std::string kopen_filename = s1;
	std::string kclose_filename = s2;
	std::ifstream in;
	in.open(kopen_filename);
	double x, y;
	std::vector<double> X, Y;
	while (in >> x >> y) {
		X.push_back(x);
		Y.push_back(y);
	}
	in.close();
	_kopen = std::make_unique<tk::spline>(X, Y);
	X.clear();
	Y.clear();
	in.open(kclose_filename);
	while (in >> x >> y) {
		X.push_back(x);
		Y.push_back(y);
	}
	in.close();
	_kclose = std::make_unique<tk::spline>(X, Y);
}

double TwoStateSplineModel::_KClose(std::vector<double>& ions) {
	return (*_kclose)(ions[0]);
}

double TwoStateSplineModel::_KOpen(std::vector<double>& ions) {
	return (*_kopen)(ions[0]);
}