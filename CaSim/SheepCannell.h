#pragma once
#include "TwoStateModel.h"
class SheepCannell : public TwoStateModel
{
private:
	double kopen0,
		ncaopen, kopenmax,
		kclose0, ncaclose;
protected:
	double _KClose(std::vector<double>&);
	double _KOpen(std::vector<double>&);
public:
	SheepCannell(std::map<std::string, double>&);
};

