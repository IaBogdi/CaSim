#pragma once
#include "TwoStatePoModel.h"
class TwoStateaHTG : public TwoStatePoModel {
	double Po(const std::unordered_map<std::string, double>&);
private:
	double fCa,
		KCa,
		fMg,
		KMg,
		KO0,
		KMgI,
		KCL,
		KI,
		KOL;
public:
	TwoStateaHTG(std::map<std::string, double>&);
};

