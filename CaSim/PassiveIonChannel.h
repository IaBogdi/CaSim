#pragma once
#include "IonChannel.h"
#include <map>
class PassiveIonChannel : public IonChannel {
	double Conductance(double, double,int);
public:
	PassiveIonChannel(std::string, std::vector<double>&, std::map<std::string, double>&,std::vector<double>&,std::string&,std::string&);
	double Flux(double, double,int);
};

