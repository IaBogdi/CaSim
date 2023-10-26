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

class CytosolRD : public CytosolSeries {
public:
	std::map<std::string, std::map<std::string, int> > idx_bufion; //index for fast detection of number for output
	std::vector < double > init_buffers;
	double R;
	std::vector<double> ions_init;
	std::vector<double> D_ions;
	int n_ions_dyad;
	int n_buffs_dyad;
	double* d_ions; //[n_elements x n_ions]
	double* d_ions_f;
	double* d_init_ions;
	double* d_D_ions;
	double* d_ions_boundary;
	double* evo_ions_diffusion;
	double* evo_ions_reaction;
	double* evo_ions_total;
	double* evo_ions;
	double* d_buffers; //[n_elements x n_buffers x n_ions]
	double* d_buffers_f;
	double* d_buffers_free;
	double* d_init_buffers;
	double* d_fluxes;
	double* d_buffers_new;
	double* d_currents;
	double* d_buffers_boundary;
	double* evo_buffers_diffusion;
	double* evo_buffers_reaction;
	double* evo_buffers_total;
	double* evo_buffers;
	int ca_idx;
	std::vector<short> buffer_binds_ion;
	std::vector<double> D_buf; //buf_idx i.e. number of ion-buffer connections
	std::vector<double> kon_buf; //buf_idx
	std::vector<double> koff_buf; //buf_idx
	std::vector<double> total_buf; // n_buffers
	std::vector<double> buf_init; // buf_idx
	std::vector < std::unique_ptr <Buffer<double> > > buffers;
	std::vector < std::unique_ptr <Ion<double> > > ions;
	std::map < std::string, std::unique_ptr < IonSERCA<double> > > ions_and_SR;
	double* d_kon;
	double* d_koff;
	double* d_D_buf;
	double* d_CTot;
	short* d_buffer_binds_ion;// [n_ions x n_buffers]
	double* d_ionsb;
	std::vector<double> Jmax;
	std::vector<double> Kup;
	std::vector<short> SR_uptakes_ion;
	short* d_SR_uptakes_ion;// [n_ions]
	double* d_Jmax;
	double* d_Kup;
	double dtt;
	std::shared_ptr<double> ions_boundary;
	std::shared_ptr<double> buffers_boundary;
	std::vector<double> currents;
	dim3 block, grid;
	double r, dr;
	double R_dyad;
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
	std::vector<double> ions_bg;
	void Initialize(nlohmann::json&, nlohmann::json&, int);
	std::vector<double> nsr_ions;
	std::vector<double> nsr_ions_w;
	std::vector<double> GaussElimination(std::vector<std::vector<double> >&);
	void _SetNumIonsandBuffsinDyad(int, int);
	std::vector<double> zero_vector;
	void _SetupStepGraph();
	bool dt_is_set;
	short* d_is_in_dyad;
	std::vector<short> is_in_dyad;
	std::vector<uint64_t> cytosol_dims;
public:
	CytosolRD(nlohmann::json&, nlohmann::json&, int);
	CytosolRD(nlohmann::json&, nlohmann::json&, nlohmann::json&, int);
	bool UsesGPU();
	void Reset();
	~CytosolRD();
	void RunRD(double, int);
	void GetIonsandBuffers(double*&,double*&);
	void Update(double*&,const std::vector<double>&);
	std::vector<double>& GetExtraCellularIons();
	virtual std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&);
	std::vector<uint64_t> GetDimensions();
};

