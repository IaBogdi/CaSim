#pragma once
#include "DyadSeries.h"

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <cmath>

#include "IonChannel.h"
#include "Structures.h"
#include "DyadChannels.h"


class DyadRD2DCPU : public DyadSeries
{
	std::unique_ptr<DyadChannels> channels;
	std::map<std::string, std::map<std::string, int> > idx_bufion; //index for fast detection of number for output
	int sr_size, sl_size;
	int n_channels;
	int ca_idx;
	std::vector < std::unique_ptr <Buffer<double> > > buffers_data;
	std::vector < std::unique_ptr <Ion<double> > > ions_data;
	std::map<std::string, std::unique_ptr<IonSarcolemma<double> > > ions_and_SL;
	std::vector < double > init_buffers;
	double R;
	std::vector<double> D_ions;
	std::vector<double> ions; //[n_elements x n_ions]
	double* init_ions;
	double* evo_ions;
	double* buffers; //[n_elements x n_buffers x n_ions]
	double* buf_free;
	double* d_buffers_near_channels;
	double* gradients;
	double* d_buffers_boundary;
	double* evo_buffers;
	double* ions_gradients;
	std::vector<short> buffer_binds_ion;
	std::vector<double> D_buf; //buf_idx i.e. number of ion-buffer connections
	std::vector<double> kon_buf; //buf_idx
	std::vector<double> koff_buf; //buf_idx
	std::vector<double> total_buf; // n_buffers
	std::vector<double> buf_init; // buf_idx
	short* d_buffer_binds_ion;// [n_ions x n_buffers]
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
	double* d_currents_grid;
	std::shared_ptr<double> ions_boundary;
	std::shared_ptr<double> buffers_boundary;
	double* currents;
	int* x_idx;
	int* y_idx;
	int* z_idx;
	double x, dx;
	double y, dy;
	double z;
	int n_elements_near_channels;
	int n_elements_thread;
	double dtt;
	int n_buffers;
	int n_buf_unique;
	int n_ions_and_buffers;
	int idx_buf;
	int n_ions;
	int nx, ny, nz;
	size_t shared_gradx;
	size_t shared_grady;
	std::vector<double> ions_near_channels; //ions near each channel
	double Cab, Mg;
	int n_threads;
	int stride;
	int n_elements;
	void _SetCurrents(const std::vector<double>&, const std::vector<double>&);
	std::vector<double> total_sr_current;
	double grad_mult;
	std::vector<double> GaussElimination(std::vector<std::vector<double> >&);
	int _Index(int, int, int);
	void _SetBoundaries();
	void _GetGradients();
	int _GetChannelIndex(int, int, int);
	double _GetIonNearChannel(int, int, int);
	std::vector<double> mult_z;
	double* ions_from_cytosol;
	double* buffers_from_cytosol;
	double sx, sy;
	bool pointers_are_set;
	bool dt_is_set;
	std::vector<uint64_t> dyad_dims;
	std::vector<uint64_t> channels_ions_dims;
	std::vector<uint64_t> channels_dims;
	const std::vector<double>* jsr_ions;
	inline double d2YdX2(double, double, double, double);
	void _EvolutionOp(int, int, int, double);
	void _UpdateEvo(int, int, int, double);
	void _GetFreeBuffers(int, int, int);
public:
	DyadRD2DCPU(nlohmann::json&, int);
	void Reset();
	~DyadRD2DCPU();
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

