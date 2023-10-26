#pragma once
#include "IonChannel.h"
class StepCurrentIonChannel : public IonChannel
{
public:
	StepCurrentIonChannel(std::vector<double>&,double, std::vector<double>&);
	double Flux(double, double, int);
};

