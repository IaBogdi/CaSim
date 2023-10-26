#include "TwoStateMWC.h"
#include <cmath>

TwoStateMWC::TwoStateMWC(std::map<std::string, double>& params) {
	fCa = params["fCa"];
	KCa = params["KCa"];
	fMg = params["fMg"];
	KMg = params["KMg"];
	KO0 = params["KO0"];
	KMgI = params["KMgI"];
	kclose = params["kclose"];
	cur_state = 0;
}

double TwoStateMWC::Po(std::vector<double>& ions)
{
	double Ca = ions[0];
	double Mg = ions[1];
	double inac = KMgI * KMgI / (KMgI * KMgI + Mg * Mg);
	double open = std::pow(Ca + KCa*fCa * (1 + Mg/(fMg*KMg)), 4);
	double close = KO0*std::pow(fCa * (Ca + KCa * (1 + Mg / KMg)), 4);
    return inac * open / (open + close);
}
