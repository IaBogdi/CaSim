#include "ParallelHybridSolver.h"

#ifdef CUDA_IS_AVAILABLE
#include <cuda_runtime_api.h> // CudaDeviceSynchronize

#endif // CUDA_IS_AVAILABLE


#include <iostream>

using json = nlohmann::json;

ParallelHybridSolver::ParallelHybridSolver(json& file) {
	time_max = file["Max Time, ms"];
	dt = file["dt, ns"];
	dt *= 1e-6;
	d_dt = dt;
	n_iter = time_max / dt;
	n_threads = file["NumBatch"];
	n_cores = file["NumCores"];
	omp_set_num_threads(n_cores);
	ca_nsr.resize(n_threads);
	ca_jsr.resize(n_threads);
	cyto.resize(n_threads);
	ca_currents.resize(n_threads);
	dyad_it = file["Iterations for Dyad"];
	idx_max.push_back(n_iter / dyad_it + 1);
	cyto_it = file["Iterations for Cytosol"];
	idx_max.push_back(n_iter / cyto_it + 1);
	sr_it = file["Iterations for SR"];
	idx_max.push_back(n_iter / sr_it + 1);
	channels_it = file["Iterations for IonChannels"];
	idx_max.push_back(n_iter / channels_it + 1);
	channels_ions_it = file["Iterations for IonsNearChannels"];
	idx_max.push_back(n_iter / channels_ions_it + 1);
	n_iterations_output = file["Time output"];
	outputs = file["Outputs"];
	experiment_name = std::string(file["Name"]);
	fs::create_directory(experiment_name);
	std::string write_pars = experiment_name + std::string("/Parameters.json");
	std::ofstream o(write_pars.c_str());
	o << std::setw(4) << file << std::endl;
	o.close();
	std::string h5filename = experiment_name + std::string("/Output.h5");
	h5file = H5::H5File(h5filename.c_str(), H5F_ACC_TRUNC);
	dyad_out = h5file.createGroup("/Dyad");
	cytosol_out = h5file.createGroup("/Cytosol");
	sr_out = h5file.createGroup("/SR");
	channels = h5file.createGroup("/Channels");
	channels_ions = h5file.createGroup("/IonsChannels");
	dyad_dataset_created = false;
	cytosol_dataset_created = false;
	sr_dataset_created = false;
	channels_dataset_created = false;
	channels_ions_dataset_created = false;
	//create compartments
	if (file["nSR"]["Type"] == "ConstPool")
		nsr = std::make_unique<NSRConstPool>(file["nSR"], n_threads);

	std::unordered_map<std::string_view, std::function<std::unique_ptr<DyadSeries>(json&,int) > > dyad_type;
	/*
	dyad_type["Reaction-Diffusion"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRD>(file["Dyad"], n_thr);
	};
	*/
#ifdef CUDA_IS_AVAILABLE
	dyad_type["3D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRDMP>(file["Dyad"], file["jSR"], n_thr);
};
	dyad_type["2D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRDMP2D>(file["Dyad"], file["jSR"], n_thr);
	};
#else
	dyad_type["3D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRDCPU>(file["Dyad"], file["jSR"], n_thr);
	};
	dyad_type["2D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRD2DCPU>(file["Dyad"], file["jSR"], n_thr);
	};
#endif // CUDA_IS_AVAILABLE
	dyad_type["CPU3D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRDCPU>(file["Dyad"], file["jSR"], n_thr);
	};
	dyad_type["CPU2D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRD2DCPU>(file["Dyad"], file["jSR"], n_thr);
	};
	/*
	dyad_type["Reaction-DiffusionCyto2D"] = [](json& file, int n_thr) {
		return std::make_unique<DyadRD2DwCytosol>(file["Dyad"], file["Cytosol"], n_thr);
	};
	*/
	std::string_view str(file["Dyad"]["Type"]);
	dyad = dyad_type[str](file,n_threads);
	dyad_dims = dyad->GetDimensions();
	channels_dims = dyad->GetChannelsDimensions();
	channels_ions_dims = dyad->GetIonsNearChannelsDimensions();
	std::unordered_map<std::string_view, std::function<std::unique_ptr<CytosolSeries>(json&, int) > > cytosol_type;
	/*
	cytosol_type["Reaction-Diffusion"] = [](json& file, int n_thr) {
		if (file["nSR"]["Type"] == "ConstPool")
			return std::make_unique<CytosolRD>(file["Cytosol"], file["Dyad"], file["nSR"], n_thr);
		return std::make_unique<CytosolRD>(file["Cytosol"], file["Dyad"], n_thr);
	};
	cytosol_type["Reaction-DiffusionCPU"] = [](json& file, int n_thr) {
		if (file["nSR"]["Type"] == "ConstPool")
			return std::make_unique<CytosolRDCPU>(file["Cytosol"], file["Dyad"], file["nSR"], n_thr);
		return std::make_unique<CytosolRDCPU>(file["Cytosol"], file["Dyad"], n_thr);
	};
	cytosol_type["Reaction-DiffusionMP"] = [](json& file, int n_thr) {
		if (file["nSR"]["Type"] == "ConstPool")
			return std::make_unique<CytosolRDMP>(file["Cytosol"], file["Dyad"], file["nSR"], n_thr);
		return std::make_unique<CytosolRDMP>(file["Cytosol"], file["Dyad"], n_thr);
	};
	*/
	cytosol_type["Pool"] = [](json& file, int n_thr) {
		if (file["nSR"]["Type"] == "ConstPool")
			return std::make_unique<CytosolPool>(file["Cytosol"], file["Dyad"], file["nSR"], n_thr);
		return std::make_unique<CytosolPool>(file["Cytosol"], file["Dyad"], n_thr);
	};
	std::string_view str2(file["Cytosol"]["Type"]);
	//cytosol = !str.compare("Reaction-DiffusionCyto2D") ? std::make_unique<EmptyCytosol>(file["Cytosol"], file["Dyad"]) : cytosol_type[str2](file, n_threads);
	cytosol = cytosol_type[str2](file, n_threads);
	cytosol_dims = cytosol->GetDimensions();
	dyad->IsCytosolGPU(cytosol->UsesGPU());
	if (file["jSR"]["Type"] == "Pool") {
		sr = std::make_unique<SRPool>(file["jSR"], n_threads, dyad->GetElementVolume(), dyad->GetNumSRChannels(), dyad->GetListofIons());
		sr_dims = sr->GetDimensions();
	}
}

void ParallelHybridSolver::RunSimulations(long long n_scenarios) {
	switch (int hash = 8 * nsr->UsesGPU() + 4 * sr->UsesGPU() + 2 * dyad->UsesGPU() + cytosol->UsesGPU(); hash) {
	case 0:
		RunSimulationsCPU(n_scenarios);
		break;
	case 2:
		RunSimulationsSRCytoPool(n_scenarios);
		break;
	case 3:
		RunSimulationsSRPool(n_scenarios);
		break;
	default:
		return;
	}
}

void ParallelHybridSolver::ResetSystem() {
#ifdef CUDA_IS_AVAILABLE
	cudaDeviceSynchronize();
#endif // CUDA_IS_AVAILABLE	
	nsr->Reset();
	sr->Reset();
	dyad->Reset();
	cytosol->Reset();
}

void ParallelHybridSolver::Update() {
	double* d_ions_cytosol = nullptr;
	double* d_buffers_cytosol = nullptr;
	double* d_currents = nullptr;
#ifdef CUDA_IS_AVAILABLE
	cudaDeviceSynchronize();
#endif // CUDA_IS_AVAILABLE	
	cytosol->GetIonsBuffersandV(d_ions_cytosol, d_buffers_cytosol,V);
	dyad->Update(d_ions_cytosol, d_buffers_cytosol, sr->GetIons(), cytosol->GetExtraCellularIons(),V);
	dyad->GetEffluxes(d_currents);
	cytosol->Update(d_currents, nsr->GetIons());
#ifdef CUDA_IS_AVAILABLE
	cudaDeviceSynchronize();
#endif // CUDA_IS_AVAILABLE	
	sr->Update(dyad->GetTotalSRCurrent(), nsr->GetIons());
	nsr->Update(d_ions_cytosol, sr->GetIons());
}

void ParallelHybridSolver::_ExtendDataset(H5::DataSet& dataset, std::vector<hsize_t>& dims,int idx_time, int idx_batch, int idx_ds) {
	if (idx_time == dims[0] && dims[0] != idx_max[idx_ds]) {
		++dims[0];
		dataset.extend(dims.data());
		return;
	}
	if (idx_batch * n_threads == dims[1]) {
		dims[1] += n_threads;
		dataset.extend(dims.data());
		return;
	}
}

/*
int ParallelHybridSolver::_FstyleIdx(std::vector<int>& idxs, std::vector<hsize_t>& dims) {
	int index = 0;
	int stride = 1;
	for (int i = 0; i < dims.size(); ++i) {
		index += stride * idxs[i];
		stride *= dims[i];
	}
	return index;
}

int ParallelHybridSolver::_CstyleIdx(std::vector<int>& idxs, std::vector<hsize_t>& dims) {
	int index = 0;
	int stride = 1;
	for (int i = dims.size() - 1; i >= 0; --i) {
		index += stride * idxs[i];
		stride *= dims[i];
	}
	return index;
}

template <typename T>
void ParallelHybridSolver::_RowToColIterator(std::vector<int>& index, std::vector<hsize_t>& dims, int dimension,
	std::vector<T>& out_arr, const std::vector<T>& in_arr) {
	if (dimension == dims.size()) {
		int index_f = _FstyleIdx(index, dims);
		int index_c = _CstyleIdx(index, dims);
		out_arr[index_c] = in_arr[index_f];
	}
	else {
		for (int i = 0; i < dims[dimension]; i++) {
			index[dimension] = i;
			_RowToColIterator(index, dims, dimension + 1, out_arr, in_arr);
		}
	}
}

template <typename T>
void ParallelHybridSolver::_ColumntoRowMajor(std::vector<T>& out_arr,const std::vector<T>& in_arr, std::vector<hsize_t>& dims) {
	std::vector<int> index(dims.size(), 0);
	_RowToColIterator(index, dims, 0, out_arr, in_arr);
}
*/

template <typename T>
void ParallelHybridSolver::_WriteDataPart(H5::Group& data_group, std::map<std::string, std::vector<T> >& data,
	const H5::PredType& type, std::vector<hsize_t>& dims, bool& dataset_created, int idx_time, int idx_batch, int idx_ds) {
	if (!dataset_created) {
		dims[0] = 1;
		std::vector<hsize_t> max_dims(dims);
		max_dims[0] = idx_max[idx_ds];
		max_dims[1] = n_scenarios;
		H5::DataSpace dataspace(dims.size(), dims.data(), max_dims.data());
		std::vector<hsize_t> chunk_dims(dims);
		H5::DSetCreatPropList prop_list;
		prop_list.setChunk(chunk_dims.size(), chunk_dims.data());
		std::vector<hsize_t> offset(dims.size(), 0);
		std::vector<hsize_t> count(dims);
		count[0] = 1;
		/*
		int total_elements = 1;
		for (int i = 0; i < count.size(); ++i)
			total_elements *= count[i];
		if (type == H5::PredType::NATIVE_DOUBLE) 
			h5data_to_write.push_back(std::vector<double>(total_elements));
		else
			h5data_to_write.push_back(std::vector<int>(total_elements));
		*/
		for (const auto& pair : data) {
			auto ds = data_group.createDataSet(pair.first.c_str(), type, dataspace, prop_list);
			// Write data to the newly created dataset.
			H5::DataSpace write_dataspace = ds.getSpace();
			write_dataspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());
			H5::DataSpace mem_space(count.size(), count.data());
			/*
			if constexpr (std::is_same_v<decltype(pair.second), std::vector<double> >) {
				_ColumntoRowMajor(std::get<std::vector<double> >(h5data_to_write[idx_ds]), pair.second, count);
				ds.write(std::get<std::vector<double> >(h5data_to_write[idx_ds]).data(), type, mem_space, write_dataspace);
				//ds.write(std::get<pair.second.data(), type, mem_space, write_dataspace);
			}
			else if constexpr (std::is_same_v<decltype(pair.second), std::vector<int> >) {
				_ColumntoRowMajor(std::get<std::vector<int> >(h5data_to_write[idx_ds]), pair.second, count);
				ds.write(std::get<std::vector<int> >(h5data_to_write[idx_ds]).data(), type, mem_space, write_dataspace);
				//ds.write(std::get < pair.second.data(), type, mem_space, write_dataspace);
			}
			*/
			ds.write(pair.second.data(), type, mem_space, write_dataspace);
			mem_space.close();
			write_dataspace.close();
			ds.close();
		}
		dataset_created = true;
		return;
	}
	for (const auto& pair : data) {
		auto openedDataset = data_group.openDataSet(pair.first.c_str());
		std::vector<hsize_t> cur_dims(dims);
		auto dataspace = openedDataset.getSpace();
		dataspace.getSimpleExtentDims(cur_dims.data());
		_ExtendDataset(openedDataset, cur_dims, idx_time, idx_batch, idx_ds);
		dataspace = openedDataset.getSpace();
		std::vector<hsize_t> offset(cur_dims.size());
		offset[0] = idx_time;
		offset[1] = cur_dims[1] - n_threads;
		std::vector<hsize_t> count(cur_dims);
		count[0] = 1;
		count[1] = n_threads;
		dataspace.selectHyperslab(H5S_SELECT_SET, count.data(), offset.data());
		H5::DataSpace mem_space(count.size(), count.data());
		/*
		if constexpr (std::is_same_v<decltype(pair.second), std::vector<double> >) {
			_ColumntoRowMajor(std::get<std::vector<double> >(h5data_to_write[idx_ds]), pair.second, count);
			openedDataset.write(std::get<std::vector<double> >(h5data_to_write[idx_ds]).data(), type, mem_space, dataspace);
		}
		else if constexpr (std::is_same_v<decltype(pair.second), std::vector<int> >) {
			_ColumntoRowMajor(std::get<std::vector<int> >(h5data_to_write[idx_ds]), pair.second, count);
			openedDataset.write(std::get<std::vector<int> >(h5data_to_write[idx_ds]).data(), type, mem_space, dataspace);
		}
		*/
		openedDataset.write(pair.second.data(), type, mem_space, dataspace);
		mem_space.close();
		dataspace.close();
		openedDataset.close();
	}
}

void ParallelHybridSolver::_WriteData(int j, int idx_batch) {
	if (j % dyad_it == 0) {
		auto d = dyad->GetConcentrations(outputs);
		_WriteDataPart<double>(dyad_out, d, H5::PredType::NATIVE_DOUBLE, dyad_dims, dyad_dataset_created, j / dyad_it, idx_batch, 0);
	}
	if (j % cyto_it == 0 && cytosol_dims.size() > 0) {
		auto d = cytosol->GetConcentrations(outputs);
		_WriteDataPart<double>(cytosol_out, d, H5::PredType::NATIVE_DOUBLE, cytosol_dims, cytosol_dataset_created, j / cyto_it, idx_batch, 1);
	}
	if (j % sr_it == 0) {
		auto d = sr->GetConcentrations(outputs);
		_WriteDataPart<double>(sr_out, d, H5::PredType::NATIVE_DOUBLE, sr_dims, sr_dataset_created, j / sr_it, idx_batch, 2);
	}
	if (j % channels_it == 0) {
		auto d = dyad->GetChannelsStates(outputs);
		_WriteDataPart<int>(channels, d, H5::PredType::NATIVE_INT, channels_dims, channels_dataset_created, j / channels_it, idx_batch, 3);
	}
	if (j % channels_ions_it == 0) {
		auto d = dyad->GetIonsNearChannels(outputs);
		_WriteDataPart<double>(channels_ions, d, H5::PredType::NATIVE_DOUBLE, channels_ions_dims, channels_ions_dataset_created, j / channels_ions_it, idx_batch, 4);
	}
}

void ParallelHybridSolver::RunSimulationsSRPool(int n_sc) {
	n_scenarios = n_sc;
	int n_batches = n_scenarios / n_threads;
	int sc_per_channel = n_batches * n_threads / dyad->GetNumSRChannels();
	if (sc_per_channel < 1)
		++sc_per_channel;
	for (int i = 0; i < n_batches; ++i) {
		ResetSystem();
		for (int j = 0; j < n_threads; ++j)
			dyad->InitOpening(j, (n_threads * i + j) /  sc_per_channel);
		Update();
		_WriteData(0, i);
		std::cout << "Batch " << i << std::endl;
		for (int j = 1; j <= n_iter; ++j) {
			//check for iterations and outputs
			//GPU iterations
			cytosol->RunRD(d_dt, 0);
			dyad->RunRD(d_dt, 0);
			//CPU iterations
#pragma omp parallel for num_threads(n_cores)
			for (int k = 0; k < n_threads; ++k) {
				sr->Run(dt, k);
				dyad->RunMC(dt, k);
			}
			Update();
			//check for iterations and outputs
			if (j % n_iterations_output == 0) {
				std::cout << j * dt << " ms" << std::endl;
			}
			_WriteData(j, i);
		}
	}
}

void ParallelHybridSolver::RunSimulationsSRCytoPool(int n_sc) {
	n_scenarios = n_sc;
	int n_batches = n_scenarios / n_threads;
	int sc_per_channel = n_batches * n_threads / dyad->GetNumSRChannels();
	if (sc_per_channel < 1)
		++sc_per_channel;
	for (int i = 0; i < n_batches; ++i) {
		ResetSystem();
		for (int j = 0; j < n_threads; ++j)
			dyad->InitOpening(j, (n_threads * i + j) / sc_per_channel);
		Update();
		_WriteData(0, i);
		std::cout << "Batch " << i << std::endl;
		for (int j = 1; j <= n_iter; ++j) {
			//check for iterations and outputs
			//GPU iterations
			dyad->RunRD(d_dt, 0);
			//CPU iterations
#pragma omp parallel for num_threads(n_cores)
			for (int k = 0; k < n_threads; ++k) {
				cytosol->RunRD(dt, k);
				sr->Run(dt, k);
				dyad->RunMC(dt, k);
			}
			Update();
			//check for iterations and outputs
			if (j % n_iterations_output == 0) {
				std::cout << j * dt << " ms" << std::endl;
			}
			_WriteData(j, i);
		}
	}
}


void ParallelHybridSolver::RunSimulationsCPU(int n_sc) {
	n_scenarios = n_sc;
	int n_batches = n_scenarios / n_threads;
	int sc_per_channel = n_batches * n_threads / dyad->GetNumSRChannels();
	if (sc_per_channel < 1)
		++sc_per_channel;
	for (int i = 0; i < n_batches; ++i) {
		ResetSystem();
		for (int j = 0; j < n_threads; ++j)
			dyad->InitOpening(j, (n_threads * i + j) / sc_per_channel);
		Update();
		_WriteData(0,i);
		std::cout << "Batch " << i << std::endl;
		for (int j = 1; j <= n_iter; ++j) {
#pragma omp parallel for num_threads(n_cores)
			for (int k = 0; k < n_threads; ++k) {
				cytosol->RunRD(d_dt, k);
				dyad->RunRD(d_dt, k);
				sr->Run(dt, k);
				dyad->RunMC(dt, k);
			}
			Update();
			//check for iterations and outputs
			if (j % n_iterations_output == 0) {
				std::cout << j * dt << " ms" << std::endl;
			}
			_WriteData(j, i);
		}
	}
}

ParallelHybridSolver::~ParallelHybridSolver() {
	sr_out.close();
	channels.close();
	channels_ions.close();
	cytosol_out.close();
	dyad_out.close();
	h5file.close();
}
