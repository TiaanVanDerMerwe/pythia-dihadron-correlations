#include "Pythia8/Pythia.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <math.h>

using namespace Pythia8;

// Stale function. Use ParticleDecays:limitTau0 instead
bool isPrimaryCharged(const Pythia8::Particle& p, const Pythia8::Event& event)
{
    // Must be final state and charged
    if (!p.isFinal() || !p.isCharged() || !p.isHadron()) return false;

    // Lifetime cut: c*tau > 1 cm
    if (p.tau0() > 0.0 && p.tau0() < 10.0) return false; // tau0 in mm → 10 mm = 1 cm

    // Walk up the ancestry chain
    int iMom = p.mother1();

    while (iMom > 0) {
        const auto& mom = event[iMom];

        // If parent is long-lived → this particle is secondary
        if (mom.tau0() > 10.0) return false;

        iMom = mom.mother1();
    }

    return true;
}

std::string makeFilename(double COM, double POW, int SEED,
                         double pTHatMin, double pTHatMax)
{
    std::ostringstream oss;
    oss << "pythiaData/" << std::to_string((int)COM)
        << "/cms/expComparison/19.2-24.0/default/dihadron_pow"
        << std::to_string((int)POW)
        << "_pT" << std::to_string((int)pTHatMin)
        << "to"  << std::to_string((int)pTHatMax)
        << "_seed" << std::to_string((int)SEED) << ".csv";
    return oss.str();
}

// Derive the particle-counts filename from the data filename
// e.g.  pythiaData/200/(cms or star or alice)/dihadron_pow2_pT10to20_seed1.csv
//    -> pythiaData/200/(cms or star or alice)/dihadron_pow2_pT10to20_seed1_particle_counts.txt
std::string makeCountFilename(const std::string& dataFilename)
{
    std::string out = dataFilename;
    // Strip ".csv" suffix (if present) then append tag
    const std::string suffix = ".csv";
    if (out.size() >= suffix.size() &&
        out.compare(out.size() - suffix.size(), suffix.size(), suffix) == 0)
        out.erase(out.size() - suffix.size());
    out += "_particle_counts.txt";
    return out;
}

int main(int argc, char* argv[]) {

    // ── Now takes 5 args: COM  nEvents  power  seed  bin_index ──────
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <COM_energy> <num_events> <power> <seed> <bin_index>\n";
        return 1;
    }

    double COM     = std::atof(argv[1]);
    int    NEVENTS = std::atoi(argv[2]);
    double POW     = std::atof(argv[3]);
    int    SEED    = std::atoi(argv[4]);
    int    iBin    = std::atoi(argv[5]);

    // ── pTHat bin edges — must match run.sh ─────────────────────────
    std::vector<double> pTlimit = {18.0, 600.0};
    int nBin = pTlimit.size() - 1;

    if (iBin < 0 || iBin >= nBin) {
        std::cerr << "bin_index " << iBin
                  << " out of range [0, " << nBin - 1 << "]\n";
        return 1;
    }

    double pTMin = pTlimit[iBin];
    double pTMax = pTlimit[iBin + 1];   // -1 = no upper limit in Pythia

    std::cout << "Bin " << iBin << ": pTHat = [" << pTMin
              << ", " << (pTMax < 0 ? std::numeric_limits<double>::infinity() : pTMax)
              << "] GeV"
              << "  |  events=" << NEVENTS << "\n";

    // ── Dihadron parameters (unchanged from your original) ───────────
    const std::vector<std::pair<double,double>> trigRanges = {
        {19.2, 24.0}
    };
    const double TRIG_PT_MIN  = trigRanges.front().first;
    const double TRIG_PT_MAX  = trigRanges.back().second;
    const double ASSOC_PT_MIN = 0.5,  ASSOC_PT_MAX = 19;
    const double TRIG_ETA_MIN = -2.0, TRIG_ETA_MAX = 2.0;
    const double ASSOC_ETA_MIN= -2.0, ASSOC_ETA_MAX= 2.0;

    // ── Pythia setup ─────────────────────────────────────────────────
    Pythia pythia;

    pythia.readString("Tune:pp = 14");
    pythia.readString("Beams:eCM = " + std::to_string(COM));
    pythia.readString("Beams:idA = 2212");
    pythia.readString("Beams:idB = 2212");

    pythia.readString("Init:showProcesses = off");
    pythia.readString("Init:showMultipartonInteractions = off");
    pythia.readString("Init:showChangedSettings = off");
    pythia.readString("Init:showChangedParticleData = off");
    pythia.readString("Next:numberCount = 1000");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(SEED));

    pythia.readString("SoftQCD:all = off");
    pythia.readString("HardQCD:all = on");
    pythia.readString("PartonLevel:MPI = on");
    pythia.readString("PartonLevel:ISR = on");
    pythia.readString("PartonLevel:FSR = on");
    pythia.readString("HadronLevel:Hadronize = on");
    pythia.readString("HadronLevel:Decay = on");

    //pythia.readString("ParticleDecays:limitTau0 = on");
    //pythia.readString("ParticleDecays:tau0Max = 10");

    pythia.readString("PhaseSpace:pTHatMin = " + std::to_string(pTMin));
    if (pTMax > 0) pythia.readString("PhaseSpace:pTHatMax = " + std::to_string(pTMax));

    pythia.readString("PhaseSpace:bias2Selection = on");
    pythia.readString("PhaseSpace:bias2SelectionPow = " + std::to_string(POW));
    pythia.readString("PhaseSpace:bias2SelectionRef = " + std::to_string(5));

    std::cout << "Seed check: " << pythia.settings.mode("Random:seed")
          << "  setSeed: "  << pythia.settings.flag("Random:setSeed") << "\n";

    if (!pythia.init()) {
        std::cerr << "Pythia init failed for bin " << iBin << "\n";
        return 1;
    }

    // ── Output file ──────────────────────────────────────────────────
    std::string fname = makeFilename(COM, POW, SEED, pTMin,
                                     pTMax < 0 ? 999 : pTMax);
    std::ofstream out(fname);
    if (!out.is_open()) {
        std::cerr << "Cannot open output file: " << fname << "\n";
        return 1;
    }
    out << std::scientific << std::setprecision(6);

    const size_t bufferSize = 64 * 1024 * 1024;
    std::vector<char> fileBuffer(bufferSize);
    out.rdbuf()->pubsetbuf(fileBuffer.data(), bufferSize);

    // ── Header ───────────────────────────────────────────────────────
    out << "# BIN: "             << iBin   << " of " << nBin   << "\n";
    out << "# NEVENTS: "         << NEVENTS                    << "\n";
    out << "# POWER: "           << POW                        << "\n";
    out << "# PREF: "            << 5                          << "\n";
    out << "# PTHAT_RANGE: "     << pTMin  << " - " << pTMax   << "\n";
    out << "# TRIG_PT_RANGE: "   << TRIG_PT_MIN  << " - " << TRIG_PT_MAX  << "\n";
    out << "# TRIG_ETA_RANGE: "  << TRIG_ETA_MIN << " - " << TRIG_ETA_MAX << "\n";
    out << "# ASSOC_PT_RANGE: "  << ASSOC_PT_MIN << " - pTtrig (dynamic)\n";
    out << "# ASSOC_ETA_RANGE: " << ASSOC_ETA_MIN<< " - " << ASSOC_ETA_MAX<< "\n";
    out << "\n# Dihadron correlation data\n";
    out << "event,weight,trigger_id,trigger_pT,trigger_eta,trigger_phi,"
        << "assoc_id,assoc_pT,assoc_eta,assoc_phi\n";

    // ── Event loop ───────────────────────────────────────────────────
    double sumWeights       = 0.0; // Sum of all event weights
    double triggerWeightSum = 0.0; // Sum of all trigger weights
    std::vector<double> rangeWeightSums(trigRanges.size(), 0.0); // Sums of trigger weights that fall in a pTtrig range (sums to triggerWeightSum)
    int globalEvent = 0; // Number of events that run
    int triggerCount = 0; // Number of triggers
    int pairCount = 0; // Number of unique trigger-associate pairs

    // ── Particle-type counters ────────────────────────────────────────
    // Key: particle name (from Pythia's particle database)
    // Value: number of times that species passed the cut
    std::map<std::string, long long> triggerParticleCounts;
    std::map<std::string, long long> assocParticleCounts;

    for (int iEvent = 0; iEvent < NEVENTS; ++iEvent) {
        if (!pythia.next()) continue;
        globalEvent++;

        double eventWeight = pythia.info.weight();
        sumWeights += eventWeight;

        std::vector<int> triggerIndices;
        for (int i = 0; i < pythia.event.size(); ++i) {
            const Particle& p = pythia.event[i];
            if (!p.isFinal() || !p.isCharged() || !p.isHadron()) continue;
            double pt = p.pT(), eta = p.eta();
            if (pt >= TRIG_PT_MIN && pt <= TRIG_PT_MAX &&
                eta >= TRIG_ETA_MIN && eta <= TRIG_ETA_MAX) {
                triggerWeightSum += eventWeight;
                triggerIndices.push_back(i);

                // ── Count by particle name ────────────────────────────
                std::string pname = pythia.particleData.name(p.id());
                triggerParticleCounts[pname]++;

                for (int r = 0; r < (int)trigRanges.size(); ++r) {
                    if (pt >= trigRanges[r].first && pt < trigRanges[r].second) {
                        rangeWeightSums[r] += eventWeight;
                        break;
                    }
                }
            }
        }

        triggerCount += triggerIndices.size();

        for (int iTrig : triggerIndices) {
            const Particle& trigger = pythia.event[iTrig];
            for (int i = 0; i < pythia.event.size(); ++i) {
                if (i == iTrig) continue;
                const Particle& assoc = pythia.event[i];
                if (!assoc.isFinal() || !assoc.isCharged() || !assoc.isHadron()) continue;

                double pt = assoc.pT(), eta = assoc.eta();
                if (pt >= ASSOC_PT_MIN && pt < ASSOC_PT_MAX &&
                    eta >= ASSOC_ETA_MIN && eta <= ASSOC_ETA_MAX) {

                        out << globalEvent   << ","
                        << eventWeight   << ","
                        << iTrig         << ","
                        << trigger.pT()  << ","
                        << trigger.eta() << ","
                        << trigger.phi() << ","
                        << i             << ","
                        << assoc.pT()    << ","
                        << assoc.eta()   << ","
                        << assoc.phi()   << "\n";
                    ++pairCount;

                    // Count associate by particle name (once per trigger–assoc pair)
                    std::string pname = pythia.particleData.name(assoc.id());
                    assocParticleCounts[pname]++;
                }
            }
        }
    }

    // ── Trailing statistics ──────────────────────────────────────────
    out << "# sigmaGEN: "         << pythia.info.sigmaGen()   << "\n";
    out << "# weightSum: "        << sumWeights         << "\n";
    out << "# nEvents  : "        << globalEvent       << "\n";
    out << "# nTriggers: "        << triggerCount       << "\n";
    out << "# nPairs   : "        << pairCount       << "\n";
    out << "# triggerWeightSum: " << triggerWeightSum   << "\n";
    for (int r = 0; r < (int)trigRanges.size(); ++r) {
        out << "# triggerWeightSum_" << trigRanges[r].first
            << "to" << trigRanges[r].second << ": "
            << rangeWeightSums[r] << "\n";
    }
    out.close();

    // ── Write particle-count file ─────────────────────────────────────
    std::string cntFname = makeCountFilename(fname);
    std::ofstream cntOut(cntFname);
    if (!cntOut.is_open()) {
        std::cerr << "Warning: cannot open particle-count file: " << cntFname << "\n";
    } else {
        // Helper: sort a map by descending count, then alphabetically
        auto sortedEntries = [](const std::map<std::string, long long>& m)
        {
            std::vector<std::pair<std::string, long long>> v(m.begin(), m.end());
            std::sort(v.begin(), v.end(),
                [](const std::pair<std::string,long long>& a,
                   const std::pair<std::string,long long>& b) {
                    return a.second != b.second ? a.second > b.second
                                                : a.first < b.first;
                });
            return v;
        };

        cntOut << "# Particle counts for bin " << iBin << " of " << nBin << "\n";
        cntOut << "# PTHAT_RANGE: " << pTMin << " - " << pTMax << "\n";
        cntOut << "# NEVENTS_GENERATED: " << globalEvent << "\n";
        cntOut << "#\n";
        cntOut << "# Trigger condition:\n";
        cntOut << "#   pT in [" << TRIG_PT_MIN << ", " << TRIG_PT_MAX << "] GeV\n";
        cntOut << "#   |eta| in [" << TRIG_ETA_MIN << ", " << TRIG_ETA_MAX << "]\n";
        cntOut << "#   final-state charged particles only\n";
        cntOut << "#\n";
        cntOut << "# Associate condition:\n";
        cntOut << "#   pT in [" << ASSOC_PT_MIN << ", pTtrig] GeV  (dynamic upper bound)\n";
        cntOut << "#   |eta| in [" << ASSOC_ETA_MIN << ", " << ASSOC_ETA_MAX << "]\n";
        cntOut << "#   final-state charged particles only\n";
        cntOut << "#   (counted once per trigger-assoc pair)\n";
        cntOut << "\n";

        // ── Trigger table ────────────────────────────────────────────
        cntOut << "=== TRIGGER particles (total: " << triggerCount << ") ===\n";
        cntOut << std::left << std::setw(30) << "particle_name"
               << std::right << std::setw(12) << "count" << "\n";
        cntOut << std::string(44, '-') << "\n";
        for (auto& [name, cnt] : sortedEntries(triggerParticleCounts))
            cntOut << std::left  << std::setw(30) << name
                   << std::right << std::setw(12) << cnt << "\n";
        cntOut << "\n";

        // ── Associate table ───────────────────────────────────────────
        long long totalAssoc = 0;
        for (auto& [n, c] : assocParticleCounts) totalAssoc += c;
        cntOut << "=== ASSOCIATE particles (total: " << totalAssoc << ") ===\n";
        cntOut << std::left << std::setw(30) << "particle_name"
               << std::right << std::setw(12) << "count" << "\n";
        cntOut << std::string(44, '-') << "\n";
        for (auto& [name, cnt] : sortedEntries(assocParticleCounts))
            cntOut << std::left  << std::setw(30) << name
                   << std::right << std::setw(12) << cnt << "\n";

        cntOut.close();
        std::cout << "  Particle counts  : " << cntFname << "\n";
    }

    std::cout << "=== Bin " << iBin << " done ===\n"
              << "  Events generated : " << globalEvent  << "\n"
              << "  Triggers found   : " << triggerCount << "\n"
              << "  Pairs written    : " << pairCount    << "\n"
              << "  Output file      : " << fname        << "\n";
    return 0;
}