#include "TwoStateaHTG.h"
#include <cmath>

TwoStateaHTG::TwoStateaHTG(std::map<std::string, double>& params) {
    fCa = params["fCa"];
    KCa = params["KCa"];
    fMg = params["fMg"];
    KMg = params["KMg"];
    KO0 = params["KO0"];
    KMgI = params["KMgI"];
    KCL = params["KCL"];
    KI = params["KI"];
    KOL = params["KOL"];
    kclose = params["kclose"];
    cur_state = 0;
}

double TwoStateaHTG::Po(const std::unordered_map<std::string, double>& ions) {
    double ca = ions.at("Calcium");
    double mg = ions.at("Magnesium");
    double mult = KMgI * KMgI / (KMgI * KMgI + mg * mg);
    double up = (KCL * KI * (std::pow(ca, 4) * std::pow(fMg, 4) * std::pow(fCa, 4) * std::pow(KMg, 4) * KO0 + std::pow(ca * fMg * KMg + fCa * (mg + fMg * KMg) * KCa, 4) * KOL));
    double down = (4 * std::pow(ca, 3) * std::pow(fMg, 3) * fCa * std::pow(KMg, 3) * KCa * KCL * KI * (mg + fMg * KMg + fMg * std::pow(fCa, 3) * (mg + KMg) * KO0) * KOL +
        6 * std::pow(ca, 2) * std::pow(fMg, 2) * std::pow(fCa, 2) * std::pow(KMg, 2) * std::pow(KCa, 2) * KCL * KI * (std::pow(mg + fMg * KMg, 2) + std::pow(fMg, 2) * std::pow(fCa, 2) * std::pow(mg + KMg, 2) * KO0) * KOL +
        4 * ca * fMg * std::pow(fCa, 3) * KMg * std::pow(KCa, 3) * KCL * KI * (std::pow(mg + fMg * KMg, 3) + std::pow(fMg, 3) * fCa * std::pow(mg + KMg, 3) * KO0) * KOL +
        std::pow(fCa, 4) * std::pow(KCa, 4) * KCL * KI * (std::pow(mg + fMg * KMg, 4) + std::pow(fMg, 4) * std::pow(mg + KMg, 4) * KO0) * KOL +
        std::pow(ca, 4) * std::pow(fMg, 4) * std::pow(KMg, 4) * (KCL * KI * KOL + std::pow(fCa, 4) * KO0 * (1 + KI + KCL * KI + KCL * KI * KOL)));
    return mult * up / down;
}
