#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <H5Cpp.h>


#include "CaSolver.h"
#include "IndChannelCREAnalyzer.h"
#include "SRPool.h"
/* 
#include "DyadRD2DwCytosol.h"
#include "DyadRD.h"
*/
#ifdef CUDA_IS_AVAILABLE
#include "DyadRDMP.h"
#include "DyadRDMP2D.h"
#endif // !CUDA_IS_AVAILABLE

#include "DyadRD2DCPU.h"
#include "DyadRDCPU.h"
/*
#include "CytosolRD.h"
#include "CytosolRDCPU.h"
#include "CytosolRDMP.h"
#include "EmptyCytosol.h"
*/
#include "CytosolPool.h"
#include "NSRConstPool.h"


namespace fs = std::filesystem;



class ParallelHybridSolver : public CaSolver {
	std::unique_ptr<NSRSeries> nsr;
	std::unique_ptr<SRSeries> sr;
	std::unique_ptr<DyadSeries> dyad;
	std::unique_ptr<CytosolSeries> cytosol;
	std::unique_ptr<IndChannelCREAnalyzer> analyzer;
	int n_threads;
	std::string experiment_name;
	double time_max;
	double dt;
	double d_dt;
	int write_data;
	int nit_print;
	long long n_iter;
	int n_cores;
	int n_scenarios;
	long dyad_it;
	long cyto_it;
	long sr_it;
	long channels_it;
	long channels_ions_it;
	int n_iterations_output;
	double V; //Voltage
	std::vector<std::vector<double> > ca_nsr;
	std::vector<std::vector<double> > ca_jsr;
	std::vector<std::vector<double> > cyto;
	std::vector<std::vector<double> > ca_currents;
	std::vector<int> idx_max;
	std::string output_cytosol;
	std::string output_dyad;
	H5::H5File h5file;
	H5::Group dyad_out;
	std::vector<hsize_t> dyad_dims;
	H5::Group cytosol_out;
	std::vector<hsize_t> cytosol_dims;
	H5::Group sr_out;
	std::vector<hsize_t> sr_dims;
	H5::Group channels;
	std::vector<hsize_t> channels_dims;
	H5::Group channels_ions;
	std::vector<hsize_t> channels_ions_dims;
	std::vector<std::string> outputs;
	// std::vector< std::variant<std::vector<int>, std::vector<double> > > h5data_to_write;
	bool dyad_dataset_created;
	bool cytosol_dataset_created;
	bool sr_dataset_created;
	bool channels_dataset_created;
	bool channels_ions_dataset_created;
	void RunSimulationsSRPool(int);
	void RunSimulationsCPU(int);
	void RunSimulationsSRCytoPool(int);
	void ResetSystem();
	void Update();
	void _WriteData(int, int);
	template <typename T>
	void _WriteDataPart(H5::Group&, std::map<std::string, std::vector<T> >&, const H5::PredType&, std::vector<hsize_t>&, bool&,int, int, int);
	void _ExtendDataset(H5::DataSet&,std::vector<hsize_t>&,int,int,int);
	/*
	template <typename T>
	void _ColumntoRowMajor(std::vector<T>&,const std::vector<T>&, std::vector<hsize_t>&);
	template <typename T>
	void _RowToColIterator(std::vector<int>&, std::vector<hsize_t>&, int, std::vector<T>&, const std::vector<T>&);
	int _FstyleIdx(std::vector<int>&, std::vector<hsize_t>&);
	int _CstyleIdx(std::vector<int>&, std::vector<hsize_t>&);
	*/
public:
	ParallelHybridSolver(nlohmann::json&);
	void RunSimulations(long long n_scenarios);
	~ParallelHybridSolver();
};

