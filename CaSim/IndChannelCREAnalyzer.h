#pragma once
#include <vector>
#include <string>

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

class IndChannelCREAnalyzer {
	double dt;
	int FindNo(std::vector<double>&,std::vector<std::vector<int> >&);
	void WriteIntoHistogram(std::vector<std::vector<std::vector<int>>>&, int, long long, long long, int);
	long long n_time_hist;
	int n_channels;
public:
	IndChannelCREAnalyzer();
	void RunAnalysis(std::string&,std::vector < std::vector < std::vector < double > > >&, std::vector < std::vector < int > >&,double,int,int);
	void CalculateGlobalHist(std::vector<std::vector<int> >&, std::vector<std::vector<std::vector<int> > >&, int,int, std::string&);
};

