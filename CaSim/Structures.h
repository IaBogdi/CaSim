#pragma once
template<typename T>
struct BufferIon {
	T D;
	T kon;
	T koff;
	T initial_C;
	BufferIon(T d, T Kon, T Koff) : D{ d }, kon{ Kon }, koff{ Koff } {
		initial_C = 0;
	};
};

template<typename T>
struct Buffer {
	std::string name;
	T Ctot;
	bool is_in_dyad;
	std::map<std::string, std::unique_ptr <BufferIon<T> > > ions_kinetics;
	Buffer(std::string Name, T ctot) : name{ Name }, Ctot{ ctot }, is_in_dyad{ false } {};
	Buffer(std::string Name, T ctot, bool isindyad) : name{ Name }, Ctot{ ctot }, is_in_dyad{ isindyad } {};
};

template<typename T>
struct Ion {
	std::string name;
	T D;
	T Cb;
	Ion(std::string Name, T d, T ctot) : name{ Name }, D{ d }, Cb{ ctot } {};
};

template<typename T>
struct IonSarcolemma {
	T N1;
	T K1;
	T N2;
	T K2;
	IonSarcolemma(T n1, T k1, T n2, T k2) : N1{ n1 }, K1{ k1 }, N2{ n2 }, K2{ k2 } {};
};

template<typename T>
struct IonSERCA {
	T Jmax;
	T Kup;
	std::string name;
	IonSERCA(std::string Name, T jmax, T kup) : name{ Name }, Jmax{ jmax }, Kup{ kup } {};
};