#pragma once
#include "KineticModel.h"
#include <map>
#include <string>


class TwoStatePoModel : public KineticModel {
protected:
	double kclose;
	virtual double Po(const std::unordered_map<std::string, double>&) = 0;
public:
	double GetRate(const std::unordered_map<std::string, double>& ions);
	TwoStatePoModel(std::map<std::string, double>&);
	TwoStatePoModel();
	bool isOpen();
	void MakeTransition(double rand);
	void SetMacroState(int,double, const std::unordered_map<std::string, double>&);
	int GetMacroState();
	int GetMacroState(int);
	int GetNStates();
};

