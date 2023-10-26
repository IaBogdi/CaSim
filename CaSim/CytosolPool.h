#pragma once
#include "CytosolSeries.h"
#include "Structures.h"

class CytosolPool : public CytosolSeries
{
public:
	std::map<std::string, std::map<std::string, int> > idx_bufion; //index for fast detection of number for output
	std::vector < double > init_buffers;
	double R;
	std::vector<double> ions_init;
	std::vector<double> D_ions;
	std::vector<double> onetau_ions;
	double onetau_coeff;
	int n_ions_dyad;
	int n_buffs_dyad;
	int idx_buf;
	int n_buf_unique;
	double* ions; //[n_elements x n_ions]
	double* ions_boundary;
	double* evo_ions;
	double* d_buffers; //[n_elements x n_buffers x n_ions]
	double* buffers;
	double* buf_free;
	double* fluxes;
	double* currents;
	double* buffers_boundary;
	double* evo_buffers;
	int ca_idx;
	std::vector<short> buffer_binds_ion;
	std::vector<double> D_buf; //buf_idx i.e. number of ion-buffer connections
	std::vector<double> onetau_buffers;
	std::vector<double> kon_buf; //buf_idx
	std::vector<double> koff_buf; //buf_idx
	std::vector<double> total_buf; // n_buffers
	std::vector<double> buf_init; // buf_idx
	std::vector < std::unique_ptr <Buffer<double> > > buffers_data;
	std::vector < std::unique_ptr <Ion<double> > > ions_data;
	std::map < std::string, std::unique_ptr < IonSERCA<double> > > ions_and_SR;
	std::vector<double> Jmax;
	std::vector<double> Kup;
	std::vector<short> SR_uptakes_ion;
	double dtt;
	double r, dr;
	double R_dyad;
	double dt;
	int n_buffers;
	long nr;
	std::vector<std::vector<double> > ions_near_channels; //ions near each channel
	int n_threads;
	long stride;
	int n_elements;
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
	bool dt_is_set;
	short* d_is_in_dyad;
	std::vector<short> is_in_dyad;
	std::vector<uint64_t> cytosol_dims;
	int n_ions;
	void _GetFreeBuffers(int);
public:
	CytosolPool(nlohmann::json&, nlohmann::json&, int);
	CytosolPool(nlohmann::json&, nlohmann::json&, nlohmann::json&, int);
	bool UsesGPU();
	void Reset();
	~CytosolPool();
	void RunRD(double, int);
	void GetIonsandBuffers(double*&, double*&);
	void Update(double*&, const std::vector<double>&);
	std::vector<double>& GetExtraCellularIons();
	virtual std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&);
	std::vector<uint64_t> GetDimensions();
};

