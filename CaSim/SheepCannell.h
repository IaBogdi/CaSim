#pragma once
#include "TwoStateModel.h"
class SheepCannell : public TwoStateModel
{
private:
	double kopen0,
		ncaopen, kopenmax,
		kclose0, ncaclose;
protected:
	double _KClose(const std::unordered_map<std::string, double>&);
	double _KOpen(const std::unordered_map<std::string, double>&);
public:
	SheepCannell(std::map<std::string, double>&);
};

