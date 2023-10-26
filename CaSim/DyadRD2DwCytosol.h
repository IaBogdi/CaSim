#pragma once
#include "DyadSeries.h"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "IonChannel.h"
#include "Structures.h"
#include "DyadChannels.h"

class DyadRD2DwCytosol : public DyadSeries {
	std::unique_ptr<DyadChannels> channels;
	std::map<std::string, std::map<std::string, int> > idx_bufion; //index for fast detection of number for output
	int sr_size, sl_size;
	int n_channels;
	int ca_idx;
	std::vector < std::unique_ptr <Buffer<float> > > buffers;
	std::vector < std::unique_ptr <Ion<float> > > ions;
	std::map<std::string, std::unique_ptr<IonSarcolemma<float> > > ions_and_SL;
	std::vector < double > init_buffers;
	float R;
	std::vector<float> D_ions;
	double* d_ions; //[n_elements x n_ions]
	float* d_ions_f;
	float* d_init_ions;
	double* d_ions_near_channels;
	float* evo_ions;
	double* d_buffers; //[n_elements x n_buffers x n_ions]
	float* d_buffers_f;
	float* d_buffers_free;
	float* d_buffers_near_channels;
	float* d_init_buffers;
	double* d_gradients_x;
	double* d_gradients_y;
	double* d_gradients;
	double* d_buffers_new;
	float* d_buffers_boundary;
	float* evo_buffers;
	float* d_ions_gradients;
	double grad_mult;
	std::vector<short> buffer_binds_ion;
	std::vector<float> D_buf; //buf_idx i.e. number of ion-buffer connections
	std::vector<float> kon_buf; //buf_idx
	std::vector<float> koff_buf; //buf_idx
	std::vector<float> total_buf; // n_buffers
	std::vector<float> buf_init; // buf_idx
	float* d_CTot;
	short* d_buffer_binds_ion;// [n_ions x n_buffers]
	short* d_SL_binds_ion;// [n_ions]
	float* d_buffers_gradients;
	std::vector<float> K1;
	std::vector<float> K2;
	std::vector<float> N1;
	std::vector<float> N2;
	std::vector<float> Jmax;
	std::vector<float> Kup;
	std::vector<short> SR_uptakes_ion;
	std::vector<short> SL_binds_ion;
	float* d_currents;
	float* d_currents_grid;
	std::shared_ptr<float> ions_boundary;
	std::shared_ptr<float> buffers_boundary;
	std::vector<float> currents;
	int blockx, blocky, blockz;
	dim3 block, grid;
	float x, dx;
	float y, dy;
	float z, dz;
	int n_elements_near_channels;
	double dtt;
	int n_buffers;
	int idx_buf;
	int n_ions;
	int nx, ny, nz;
	size_t shared_gradx;
	size_t shared_grady;
	std::vector<int> x_idx_v, y_idx_v;
	std::vector<double> ions_near_channels; //ions near each channel
	double Cab, Mg;
	int n_threads;
	int stride;
	int n_blocks_init;
	int n_blocks_currents;
	int threads_per_block;
	int n_elements;
	void _SetCurrents(const std::vector<double>&, const std::vector<double>&);
	std::vector<double> total_sr_current;
	std::vector<double> GaussElimination(std::vector<std::vector<double> >&);
	std::map<std::string, std::unique_ptr<IonSERCA<float> > > ions_and_SR;
	void _SetupUpdateGraph();
	void _SetupStepGraph();
	void _ZeroCurrents();
	std::vector<float> extracell_ions;
	std::vector<float> ions_bg;
	int n_threads_grad_x;
	std::vector<short> is_in_dyad;
	int n_threads_grad_y;
	double* ions_from_cytosol;
	double* buffers_from_cytosol;
	bool pointers_are_set;
	bool dt_is_set;
	double* d_mult_z;
	std::vector<uint64_t> dyad_dims;
	std::vector<uint64_t> channels_ions_dims;
	std::vector<uint64_t> channels_dims;
public:
	DyadRD2DwCytosol(nlohmann::json&, nlohmann::json&, int);
	void Reset();
	~DyadRD2DwCytosol();
	bool UsesGPU();
	int GetNIons();
	void InitOpening(int, int);
	void RunRD(double, int);
	void RunMC(double, int);
	void Update(double*&, double*&, const std::vector<double>&, const std::vector<double>&);
	double GetElementVolume();
	int GetNumSRChannels();
	std::vector<double> GetTotalSRCurrent();
	void GetEffluxes(double*&);
	double GetL();
	std::map < std::string, std::vector<double> > GetConcentrations(std::vector<std::string>&);
	std::map <std::string, std::vector<int> > GetChannelsStates(std::vector<std::string>&);
	std::map <std::string, std::vector<double> > GetIonsNearChannels(std::vector<std::string>&);
	std::vector<uint64_t> GetDimensions();
	std::vector<uint64_t> GetChannelsDimensions();
	std::vector<uint64_t> GetIonsNearChannelsDimensions();
};

