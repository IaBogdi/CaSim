#pragma once

#include "rngstream.h"
#include <memory>
#include <vector>

class ParallelRng
{
	std::vector<RngStream> myRngs;
	void _SetRandomSeed();
	unsigned long int seed[6];
public:
	ParallelRng(int);
	ParallelRng(unsigned long int*);
	ParallelRng();
	double runif();
	double rexp(double theta);
	void WriteSeed(std::string);
};

