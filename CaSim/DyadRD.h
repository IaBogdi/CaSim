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

class DyadRD : public DyadSeries {
	std::unique_ptr<DyadChannels> channels;
	std::map<std::string, std::map<std::string, int> > idx_bufion; //index for fast detection of number for output
	int sr_size, sl_size;
	int n_channels;
	int ca_idx;
	std::vector < std::unique_ptr <Buffer<double> > > buffers;
	std::vector < std::unique_ptr <Ion<double> > > ions;
	std::map<std::string, std::unique_ptr<IonSarcolemma<double> > > ions_and_SL;
	std::vector < double > init_buffers;
	double R;
	std::vector<double> D_ions;
	double* d_ions; //[n_elements x n_ions]
	double* d_ions_f;
	double* d_mult_z;
	double* d_init_ions;
	double* d_ions_near_channels;
	double* d_D_ions;
	double* evo_ions_diffusion;
	double* evo_ions_reaction;
	double* evo_ions_total;
	double* evo_ions;
	double* d_buffers; //[n_elements x n_buffers x n_ions]
	double* d_buffers_f;
	double* d_buffers_free;
	double* d_buffers_near_channels;
	double* d_init_buffers;
	double* d_gradients_x;
	double* d_gradients_y;
	double* d_gradients;
	double* d_buffers_new;
	double* d_buffers_boundary;
	double* evo_buffers_diffusion;
	double* evo_buffers_reaction;
	double* evo_buffers_total;
	double* evo_buffers;
	double* d_ions_gradients;
	std::vector<short> buffer_binds_ion;
	std::vector<double> D_buf; //buf_idx i.e. number of ion-buffer connections
	std::vector<double> kon_buf; //buf_idx
	std::vector<double> koff_buf; //buf_idx
	std::vector<double> total_buf; // n_buffers
	std::vector<double> buf_init; // buf_idx
	double* d_kon;
	double* d_koff;
	double* d_D_buf;
	double* d_CTot;
	short* d_buffer_binds_ion;// [n_ions x n_buffers]
	short* d_SL_binds_ion;// [n_ions]
	double* d_buffers_gradients;
	std::vector<double> K1;
	std::vector<double> K2;
	std::vector<double> N1;
	std::vector<double> N2;
	double* d_K1;
	double* d_K2;
	double* d_N1;
	double* d_N2;
	std::vector<short> SL_binds_ion;
	double* d_currents;
	double* d_currents_grid;
	std::shared_ptr<double> ions_boundary;
	std::shared_ptr<double> buffers_boundary;
	std::vector<double> currents;
	int blockx, blocky, blockz;
	dim3 block, grid;
	int* x_idx;
	int* y_idx;
	int* z_idx;
	double x, dx;
	double y, dy;
	double z, dz;
	int n_elements_near_channels;
	double dtt;
	int n_buffers;
	int idx_buf;
	int n_ions;
	int nx, ny, nz;
	size_t shared_gradx;
	size_t shared_grady;
	std::vector<int> x_idx_v, y_idx_v, z_idx_v;
	std::vector<double> ions_near_channels; //ions near each channel
	double Cab, Mg;
	int n_threads;
	int stride;
	int n_blocks_init;
	int n_blocks_currents;
	int threads_per_block;
	int n_elements;
	void _SetCurrents(const std::vector<double>&,const std::vector<double>&);
	std::vector<double> total_sr_current;
	double grad_mult;
	std::vector<double> GaussElimination(std::vector<std::vector<double> >&);
	void _SetupUpdateGraph();
	void _SetupStepGraph();
	void _ZeroCurrents();
	int n_threads_grad_x;
	int n_threads_grad_y;
	double* ions_from_cytosol;
	double* buffers_from_cytosol;
	bool pointers_are_set;
	bool dt_is_set;
	std::vector<uint64_t> dyad_dims;
	std::vector<uint64_t> channels_ions_dims;
	std::vector<uint64_t> channels_dims;
public:
	DyadRD(nlohmann::json&, int);
	void Reset();
	~DyadRD();
	bool UsesGPU();
	int GetNIons();
	void InitOpening(int, int);
	void RunRD(double, int);
	void RunMC(double,int);
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

