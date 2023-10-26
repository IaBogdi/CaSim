#include "IndChannelCREAnalyzer.h"
#include <omp.h>
#include <fstream>
#include <iostream>

IndChannelCREAnalyzer::IndChannelCREAnalyzer() {
	dt = 0.01;
}

int IndChannelCREAnalyzer::FindNo(std::vector<double>& data, std::vector<std::vector<int> >& channels_data) {
	int ans = 0;
	for (int i = 1; i < data.size(); ++i) {
		ans += channels_data[i - 1][data[i]];
	}
	return ans;
}

void IndChannelCREAnalyzer::WriteIntoHistogram(std::vector<std::vector<std::vector<int> > >& hist, int cur_thread, long long cur_idx, long long end_idx,int nopen) {
	for (int k = cur_idx; k <= end_idx; ++k)
		++hist[cur_thread][k][nopen];
	return;
}

void IndChannelCREAnalyzer::CalculateGlobalHist(std::vector<std::vector<int> > &global_hist, std::vector<std::vector<std::vector<int> > > &local_hist, int n_scenarios, int n_threads, std::string &filename) {
	for (int i = 0; i < n_threads; ++i) {
		for (long long j = 0; j <= n_time_hist; ++j) {
			for (int k = 0; k < global_hist[j].size(); ++k) {
				global_hist[j][k] += local_hist[i][j][k];
			}
		}
	}
	for (int i = 0; i <= n_time_hist; ++i) {
		int total = 0;
		for (int k = 1; k < global_hist[i].size(); ++k) {
			total += global_hist[i][k];
		}
		global_hist[i][0] = n_scenarios - total;
	}
	std::ofstream out(filename);
	for (long long i = 0; i <= n_time_hist; ++i) {
		out << i * dt << " ";
		for (int j = 0; j < global_hist[i].size(); ++j)
			out << global_hist[i][j] << " ";
		out << std::endl;
	}
	out.close();
}

/*
For each trace calculate (in parallel) Amp, TTP, TTE, integrate N*dt, (integrate N*dt)/T. Then, calculate a time-dependent histogram of open channels for all traces.
Make this stuff in parallel:Use thread private histogram for thread local computation and then add them up.
*/
void IndChannelCREAnalyzer::RunAnalysis(std::string& experiment_name,std::vector<std::vector<std::vector<double> > >& data, std::vector<std::vector<int> >& channels_data, double max_time,int n_threads,int nadir) {
	int n_scenarios = data.size();
	int n_quarks = 0, n_blips = 0, n_sparks = 0;
	n_channels = channels_data.size();
	std::vector<std::vector<double> > output_data (n_scenarios,std::vector<double>(5));
	double Max_time = max_time;
	n_time_hist = Max_time / dt;
	std::vector<std::vector<int> > global_time_histogram(n_time_hist + 1,std::vector<int>(n_channels + 1,0));
	std::vector<std::vector<int> > global_time_histogram_quark(n_time_hist + 1, std::vector<int>(2, 0));
	std::vector<std::vector<int> > global_time_histogram_blip(n_time_hist + 1, std::vector<int>(nadir + 1, 0));
	std::vector<std::vector<int> > global_time_histogram_spark(n_time_hist + 1, std::vector<int>(n_channels + 1, 0));
	std::vector<std::vector<std::vector<int> > > local_time_histogram(n_threads, std::vector<std::vector<int> >(n_time_hist + 1, std::vector<int>(n_channels + 1,0)));
	std::vector<std::vector<std::vector<int> > > local_time_histogram_quark(n_threads, std::vector<std::vector<int> >(n_time_hist + 1, std::vector<int>(2, 0)));
	std::vector<std::vector<std::vector<int> > > local_time_histogram_blip(n_threads, std::vector<std::vector<int> >(n_time_hist + 1, std::vector<int>(nadir + 1, 0)));
	std::vector<std::vector<std::vector<int> > > local_time_histogram_spark(n_threads, std::vector<std::vector<int> >(n_time_hist + 1, std::vector<int>(n_channels + 1, 0)));
	#pragma omp parallel for num_threads(n_threads)
	for (int i = 0; i < n_scenarios; ++i) {
		double amp = 0, ttp = 0, tte = 0, Q_total = 0, q_aver = 0;
		long long cur_idx = 0;
		int cur_thread = omp_get_thread_num();
		for (int j = 0; j < data[i].size(); ++j) {
			int cur_nopen = FindNo(data[i][j], channels_data);
			if (cur_nopen > amp) {
				amp = cur_nopen;
				ttp = data[i][j][0];
			}
			int next = min(j + 1, int(data[i].size() - 1));
			Q_total += cur_nopen * (data[i][next][0] - data[i][j][0]);
			long long next_idx = data[i][next][0] / dt;
			WriteIntoHistogram(local_time_histogram, cur_thread, cur_idx, min(next_idx, n_time_hist),cur_nopen);
			cur_idx = next_idx + 1;
		}
		cur_idx = 0;
		if (amp == 1) {
			++n_quarks;
			for (int j = 0; j < data[i].size(); ++j) {
				int cur_nopen = FindNo(data[i][j], channels_data);
				int next = min(j + 1, int(data[i].size() - 1));
				long long next_idx = data[i][next][0] / dt;
				WriteIntoHistogram(local_time_histogram_quark, cur_thread, cur_idx, min(next_idx, n_time_hist), cur_nopen);
				cur_idx = next_idx + 1;
			}
		}
		else if (amp <= nadir) {
			++n_blips;
			for (int j = 0; j < data[i].size(); ++j) {
				int cur_nopen = FindNo(data[i][j], channels_data);
				int next = min(j + 1, int(data[i].size() - 1));
				long long next_idx = data[i][next][0] / dt;
				WriteIntoHistogram(local_time_histogram_blip, cur_thread, cur_idx, min(next_idx, n_time_hist), cur_nopen);
				cur_idx = next_idx + 1;
			}
		}
		else {
			++n_sparks;
			for (int j = 0; j < data[i].size(); ++j) {
				int cur_nopen = FindNo(data[i][j], channels_data);
				int next = min(j + 1, int(data[i].size() - 1));
				long long next_idx = data[i][next][0] / dt;
				WriteIntoHistogram(local_time_histogram_spark, cur_thread, cur_idx, min(next_idx, n_time_hist), cur_nopen);
				cur_idx = next_idx + 1;
			}
		}
		tte = data[i].back()[0];
		q_aver = Q_total / tte;
		output_data[i] = {amp, ttp, tte, Q_total, q_aver};
	}
	auto s = experiment_name + std::string("/analysis/scenarios.txt");
	std::ofstream out(s);
	//merge histograms and write analysis table and histogram
	for (int i = 0; i < n_scenarios; ++i) {
		for (int j = 0; j < 5; ++j)
			out << output_data[i][j] << " ";
		out << std::endl;
	}
	out.close();
	s = experiment_name + std::string("/analysis/time_hist.txt");
	CalculateGlobalHist(global_time_histogram, local_time_histogram, n_scenarios, n_threads, s);
	s = experiment_name + std::string("/analysis/time_hist_quarks.txt");
	CalculateGlobalHist(global_time_histogram_quark, local_time_histogram_quark, n_quarks, n_threads, s);
	s = experiment_name + std::string("/analysis/time_hist_blips.txt");
	CalculateGlobalHist(global_time_histogram_blip, local_time_histogram_blip, n_blips, n_threads, s);
	s = experiment_name + std::string("/analysis/time_hist_sparks.txt");
	CalculateGlobalHist(global_time_histogram_spark, local_time_histogram_spark, n_sparks, n_threads, s);
	return;
}
