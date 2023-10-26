#include "ParallelRng.h"
#include <omp.h>
#include <random>
#include <fstream>

ParallelRng::ParallelRng(int Nthreads) {
	_SetRandomSeed();
	myRngs.resize(Nthreads);
}

ParallelRng::ParallelRng() {
	_SetRandomSeed();
	myRngs.resize(omp_get_num_procs());
}

ParallelRng::ParallelRng(unsigned long int* Seed) {
	RngStream::SetPackageSeed(Seed);
	for (int i = 0; i < 6; ++i)
		seed[i] = Seed[i];
	myRngs.resize(omp_get_num_procs());
}

void ParallelRng::_SetRandomSeed() {
	std::random_device rd;
	std::mt19937 engine(rd());
	std::uniform_int_distribution<unsigned long int> distribution(std::numeric_limits<unsigned long int>::min(), 
		std::numeric_limits<unsigned long int>::max());
	for (int i = 0; i < 6; ++i)
		seed[i] = distribution(engine);
	RngStream::SetPackageSeed(seed);
}

double ParallelRng::runif() {
	return(myRngs[omp_get_thread_num()].RandU01());
}

double ParallelRng::rexp(double theta) {
	return -std::log(runif()) * theta;
}

void ParallelRng::WriteSeed(std::string s) {
	std::ofstream out(s);
	for (int i = 0; i < 6; ++i)
		out << seed[i] << std::endl;
	out.close();
}