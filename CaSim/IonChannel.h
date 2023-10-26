#pragma once

#include <memory>
#include <string>

#include "KineticModel.h"
#include "Point.h"

class IonChannel {
protected:
	std::unique_ptr<Point> coordinates;
	std::unique_ptr<KineticModel> model;
	std::vector<double> g;
public:
	virtual double Flux(double, double,int) = 0;
	Point* GetCoordinates();
	KineticModel* GetKineticModel();
};

