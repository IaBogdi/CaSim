#pragma once
#include "KineticModel.h"
#include <map>
#include <string>


class TwoStatePoModel : public KineticModel {
protected:
	double kclose;
	virtual double Po(std::vector<double>&) = 0;
public:
	double GetRate(std::vector<double>& ions);
	TwoStatePoModel(std::map<std::string, double>&);
	TwoStatePoModel();
	bool isOpen();
	void MakeTransition(double rand);
	void SetMacroState(int,double,std::vector<double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

