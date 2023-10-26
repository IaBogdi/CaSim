#pragma once

#pragma once
#include "CytosolSeries.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <vector>
#include <string>
#include <map>

#include "Structures.h"
#include "DyadSeries.h"

class CytosolRDMP : public CytosolSeries {
public:
	std::map<std::string, std::map<std::string, int> > idx_bufion; //index for fast detection of number for output
	std::vector < double > init_buffers;
	double R;
	std::vector<double> ions_init;
	std::vector<float> D_ions;
	int n_ions_dyad;
	int n_buffs_dyad;
	double* d_ions; //[n_elements x n_ions]
	float* d_ions_f;
	float* d_init_ions;
	float* d_D_ions;
	double* d_ions_boundary;
	float* evo_ions_diffusion;
	float* evo_ions_reaction;
	float* evo_ions_total;
	float* evo_ions;
	double* d_buffers; //[n_elements x n_buffers x n_ions]
	float* d_buffers_f;
	float* d_buffers_free;
	float* d_init_buffers;
	float* d_fluxes;
	double* d_currents;
	double* d_buffers_boundary;
	float* evo_buffers_diffusion;
	float* evo_buffers_reaction;
	float* evo_buffers_total;
	float* evo_buffers;
	int ca_idx;
	std::vector<short> buffer_binds_ion;
	std::vector<float> D_buf; //buf_idx i.e. number of ion-buffer connections
	std::vector<float> kon_buf; //buf_idx
	std::vector<float> koff_buf; //buf_idx
	std::vector<float> total_buf; // n_buffers
	std::vector<float> buf_init; // buf_idx
	std::vector < std::unique_ptr <Buffer<float> > > buffers;
	std::vector < std::unique_ptr <Ion<float> > > ions;
	std::map<std::string, std::unique_ptr<IonSERCA<float> > > ions_and_SR;
	float* d_kon;
	float* d_koff;
	float* d_D_buf;
	float* d_CTot;
	short* d_buffer_binds_ion;// [n_ions x n_buffers]
	float* d_ionsb;
	std::vector<float> Jmax;
	std::vector<float> Kup;
	std::vector<short> SR_uptakes_ion;
	short* d_SR_uptakes_ion;// [n_ions]
	float* d_Jmax;
	float* d_Kup;
	float dtt;
	std::shared_ptr<float> ions_boundary;
	std::shared_ptr<float> buffers_boundary;
	std::vector<float> currents;
	dim3 block, grid;
	float r, dr;
	float R_dyad;
	double dt;
	int n_buffers;
	long nr;
	std::vector<std::vector<double> > ions_near_channels; //ions near each channel
	int n_threads;
	long stride;
	long long n_blocks_init;
	long long n_blocks_currents;
	int threads_per_block = 256;
	int n_elements;
	std::vector<cudaStream_t> streams;
	int ions_dyad;
	int buffers_dyad;
	std::vector<double> extracell_ions; //extracellular Ca
	std::vector<float> ions_bg;
	void Initialize(nlohmann::json&, nlohmann::json&, int);
	std::vector<float> nsr_ions;
	std::vector<float> nsr_ions_w;
	std::vector<double> GaussElimination(std::vector<std::vector<double> >&);
	void _SetNumIonsandBuffsinDyad(int, int);
	std::vector<double> zero_vector;
	void _SetupStepGraph();
	bool dt_is_set;
	short* d_is_in_dyad;
	std::vector<short> is_in_dyad;
	std::vector<uint64_t> cytosol_dims;
public:
	CytosolRDMP(nlohmann::json&, nlohmann::json&, int);
	CytosolRDMP(nlohmann::json&, nlohmann::json&, nlohmann::json&, int);
	bool UsesGPU();
	void Reset();
	~CytosolRDMP();
	void RunRD(double,int);
	void GetIonsandBuffers(double*&, double*&);
	void Update(double*&, const std::vector<double>&);
	std::vector<double>& GetExtraCellularIons();
	virtual std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&);
	std::vector<uint64_t> GetDimensions();
};
