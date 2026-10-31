// SPDX-License-Identifier: GPL-2.0-only
//
// Randomised-arbiter logging on a 5G-LENA multi-cell downlink, for the PMRT transfer study.
// A rows x cols grid of macro gNBs, per-cell Tx-power offset knob ("ptx"), two xApps
// (PowerES lowers power at low load, Coverage raises it on SLA violations) and a
// conflict-mitigation arbiter that draws accept / half / reject for each request from the
// logged policy pi0 and records the propensities. Outputs <out>/series.csv (per cell per
// second) and <out>/units.csv (one row per decision unit).

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OranPmrtScenario");

namespace
{

struct CellSec
{
    double rxBytes{0};
    double delaySum{0};
    uint64_t pkts{0};
    double usedReg{0};
    double availReg{0};
};

struct UeSec
{
    double rxBytes{0};
    double offeredBytes{0};
};

struct Unit
{
    double t0;
    uint32_t cell;
    std::string xapp;
    double cur;
    double prop;
    std::string mode;
    double applied;
};

struct World
{
    uint32_t nCells{0};
    std::vector<uint32_t> ueCell;
    std::vector<double> ueRateBps;
    std::vector<Ptr<UdpClient>> clients;
    std::vector<Ptr<NrGnbPhy>> phys;
    std::vector<double> ptx;
    std::vector<double> loadMult;
    std::vector<CellSec> cellSec;
    std::vector<UeSec> ueSec;
    std::vector<std::vector<double>> utilHist;
    std::vector<std::vector<double>> violHist;
    std::vector<Unit> units;
    std::ofstream series;
    std::mt19937_64 designRng;
    std::mt19937_64 loadRng;
    std::normal_distribution<double> gauss{0.0, 1.0};
    double baseTxDbm{43};
    double ptxMin{-9};
    double ptxMax{3};
    double stepDb{3};
    double pktBytes{1200};
    double violRatio{0.9};
    double ouTau{30};
    double ouSigma{0.5};
    double pAccept{0.5};
    double pHalf{0.2};
    double pReject{0.3};
    double esUtil{0.30};
    double covViol{0.05};
    double warmup{20};
    double period{10};
    bool placebo{false};
    int64_t gtCell{-1};
    double gtT0{-1};
    std::string gtMode;
    uint32_t pktSize{1200};
};

World w;

void
SetPtx(uint32_t c, double offset)
{
    w.ptx[c] = offset;
    w.phys[c]->SetTxPower(w.baseTxDbm + offset);
}

void
SetRates(uint32_t c)
{
    for (uint32_t u = 0; u < w.ueCell.size(); ++u)
    {
        if (w.ueCell[u] != c)
        {
            continue;
        }
        double bps = std::max(1e3, w.ueRateBps[u] * w.loadMult[c]);
        w.clients[u]->SetAttribute("Interval", TimeValue(Seconds(w.pktSize * 8.0 / bps)));
    }
}

void
UeRx(uint32_t u, Ptr<const Packet> p, const Address&, const Address&)
{
    SeqTsHeader h;
    p->PeekHeader(h);
    uint32_t c = w.ueCell[u];
    w.cellSec[c].rxBytes += p->GetSize();
    w.cellSec[c].delaySum += (Simulator::Now() - h.GetTs()).GetSeconds();
    w.cellSec[c].pkts++;
    w.ueSec[u].rxBytes += p->GetSize();
}

void
SlotStats(const SfnSf&,
          uint32_t,
          uint32_t usedReg,
          uint32_t,
          uint32_t availableRb,
          uint32_t availableSym,
          uint16_t,
          uint16_t cellId)
{
    // cellId is 1-based in assignment order.
    uint32_t c = cellId - 1;
    if (c < w.nCells)
    {
        w.cellSec[c].usedReg += usedReg;
        w.cellSec[c].availReg += double(availableRb) * availableSym;
    }
}

std::string
DrawMode()
{
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(w.designRng);
    if (r < w.pAccept)
    {
        return "accept";
    }
    if (r < w.pAccept + w.pHalf)
    {
        return "half";
    }
    return "reject";
}

void
Decide(uint32_t c)
{
    double now = Simulator::Now().GetSeconds();
    uint32_t n = std::min<uint32_t>(uint32_t(w.period), w.utilHist[c].size());
    double util = 0;
    double viol = 0;
    for (uint32_t i = w.utilHist[c].size() - n; i < w.utilHist[c].size(); ++i)
    {
        util += w.utilHist[c][i] / n;
        viol += w.violHist[c][i] / n;
    }
    double cur = w.ptx[c];
    std::string xapp;
    double prop = cur;
    if (viol > w.covViol && cur < w.ptxMax)
    {
        xapp = "Coverage";
        prop = std::min(cur + w.stepDb, w.ptxMax);
    }
    else if (util < w.esUtil && cur > w.ptxMin)
    {
        xapp = "PowerES";
        prop = std::max(cur - w.stepDb, w.ptxMin);
    }
    // The design draw happens for every opened unit, also in warm-up and placebo runs, so that
    // the design stream stays aligned across arms of one seed.
    std::string mode = xapp.empty() ? "" : DrawMode();
    bool logged = !xapp.empty() && now >= w.warmup;
    std::string applyMode = mode;
    if (now < w.warmup || w.placebo || w.gtCell >= 0)
    {
        applyMode = "accept";
    }
    if (w.gtCell == int64_t(c) && std::abs(now - w.gtT0) < 1e-6)
    {
        applyMode = w.gtMode;
    }
    if (!xapp.empty())
    {
        double target = cur;
        if (applyMode == "accept")
        {
            target = prop;
        }
        else if (applyMode == "half")
        {
            target = cur + 0.5 * (prop - cur);
        }
        SetPtx(c, target);
        if (logged)
        {
            w.units.push_back({now, c, xapp, cur, prop, mode, target});
        }
    }
    Simulator::Schedule(Seconds(w.period), &Decide, c);
}

void
Tick()
{
    double now = Simulator::Now().GetSeconds();
    double a = std::exp(-1.0 / w.ouTau);
    for (uint32_t c = 0; c < w.nCells; ++c)
    {
        uint32_t nUe = 0;
        uint32_t nViol = 0;
        for (uint32_t u = 0; u < w.ueCell.size(); ++u)
        {
            if (w.ueCell[u] != c)
            {
                continue;
            }
            nUe++;
            if (w.ueSec[u].offeredBytes > 0 &&
                w.ueSec[u].rxBytes < w.violRatio * w.ueSec[u].offeredBytes)
            {
                nViol++;
            }
        }
        const CellSec& s = w.cellSec[c];
        double util = s.availReg > 0 ? s.usedReg / s.availReg : 0.0;
        double pOut = 20.0 * util * std::pow(10.0, w.ptx[c] / 10.0);
        double energy = 130.0 + 4.7 * pOut;
        double delay = s.pkts > 0 ? 1e3 * s.delaySum / s.pkts : 0.0;
        w.series << now << ',' << c << ',' << w.ptx[c] << ',' << w.loadMult[c] << ','
                 << s.rxBytes * 8e-6 << ',' << nViol << ',' << nUe << ',' << util << ','
                 << energy << ',' << delay << '\n';
        w.utilHist[c].push_back(util);
        w.violHist[c].push_back(nUe > 0 ? double(nViol) / nUe : 0.0);
        w.cellSec[c] = CellSec{};

        double l = std::log(w.loadMult[c]);
        l = a * l + w.ouSigma * std::sqrt(1 - a * a) * w.gauss(w.loadRng);
        w.loadMult[c] = std::exp(l);
        SetRates(c);
    }
    for (uint32_t u = 0; u < w.ueCell.size(); ++u)
    {
        w.ueSec[u].rxBytes = 0;
        w.ueSec[u].offeredBytes =
            std::max(1e3, w.ueRateBps[u] * w.loadMult[w.ueCell[u]]) / 8.0;
    }
    Simulator::Schedule(Seconds(1), &Tick);
}

} // namespace

int
main(int argc, char* argv[])
{
    uint32_t rows = 3;
    uint32_t cols = 3;
    uint32_t uePerCell = 6;
    double isd = 300;
    double rateMbps = 3.0;
    double simTime = 200;
    double bandwidth = 20e6;
    double frequency = 3.5e9;
    uint32_t seed = 1;
    std::string out = "out";
    std::string channelModel = "TwoRay";

    CommandLine cmd(__FILE__);
    cmd.AddValue("rows", "Grid rows", rows);
    cmd.AddValue("cols", "Grid columns", cols);
    cmd.AddValue("uePerCell", "UEs dropped per cell", uePerCell);
    cmd.AddValue("isd", "Inter-site distance [m]", isd);
    cmd.AddValue("bandwidth", "Carrier bandwidth [Hz]", bandwidth);
    cmd.AddValue("rateMbps", "Mean offered DL rate per UE [Mb/s]", rateMbps);
    cmd.AddValue("simTime", "Episode length [s]", simTime);
    cmd.AddValue("warmup", "Warm-up: requests accepted, no units logged [s]", w.warmup);
    cmd.AddValue("period", "Decision period per cell [s]", w.period);
    cmd.AddValue("seed", "Episode seed (RngRun and design stream)", seed);
    cmd.AddValue("placebo", "Draw and log modes but apply accept", w.placebo);
    cmd.AddValue("gtCell", "Knockout cell (-1: off)", w.gtCell);
    cmd.AddValue("gtT0", "Knockout unit opening time [s]", w.gtT0);
    cmd.AddValue("gtMode", "Knockout mode: accept | reject", w.gtMode);
    cmd.AddValue("ouSigma", "Stationary sd of the log load multiplier", w.ouSigma);
    cmd.AddValue("channelModel", "Fading model: ThreeGpp | TwoRay", channelModel);
    cmd.AddValue("out", "Output directory", out);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);
    w.designRng.seed(seed * 1000003ULL + 6612);
    w.loadRng.seed(seed * 1000003ULL + 6618);
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(2000000));

    uint32_t nCells = rows * cols;
    w.nCells = nCells;
    NodeContainer gnbs;
    gnbs.Create(nCells);
    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    std::vector<Vector> sites;
    Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
    for (uint32_t r = 0; r < rows; ++r)
    {
        for (uint32_t k = 0; k < cols; ++k)
        {
            sites.emplace_back(k * isd, r * isd, 25);
            gnbPos->Add(sites.back());
        }
    }
    mob.SetPositionAllocator(gnbPos);
    mob.Install(gnbs);

    NodeContainer ues;
    ues.Create(nCells * uePerCell);
    Ptr<UniformRandomVariable> uni = CreateObject<UniformRandomVariable>();
    uni->SetStream(1000);
    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < ues.GetN(); ++i)
    {
        const Vector& b = sites[i / uePerCell];
        double x = b.x + uni->GetValue(-isd / 2, isd / 2);
        double y = b.y + uni->GetValue(-isd / 2, isd / 2);
        if (std::hypot(x - b.x, y - b.y) < 20)
        {
            x = b.x + 20;
        }
        uePos->Add(Vector(x, y, 1.5));
    }
    mob.SetPositionAllocator(uePos);
    mob.Install(ues);

    Ptr<NrPointToPointEpcHelper> epc = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> bf = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nr = CreateObject<NrHelper>();
    nr->SetBeamformingHelper(bf);
    nr->SetEpcHelper(epc);
    nr->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerOfdmaPF"));
    nr->SetSchedulerAttribute("EnableSrsInUlSlots", BooleanValue(false));
    nr->SetSchedulerAttribute("EnableSrsInFSlots", BooleanValue(false));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(frequency, bandwidth, 1);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    Ptr<NrChannelHelper> ch = CreateObject<NrChannelHelper>();
    ch->ConfigureFactories("UMa", "Default", channelModel);
    ch->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
    ch->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    ch->AssignChannelsToBands({band});
    BandwidthPartInfoPtrVector bwps = CcBwpCreator::GetAllBwps({band});

    bf->SetAttribute("BeamformingMethod", TypeIdValue(DirectPathBeamforming::GetTypeId()));
    epc->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));
    nr->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    nr->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetGnbAntennaAttribute("AntennaElement",
                               PointerValue(CreateObject<IsotropicAntennaModel>()));
    nr->SetGnbPhyAttribute("Numerology", UintegerValue(0));
    nr->SetGnbPhyAttribute("TxPower", DoubleValue(w.baseTxDbm));

    NetDeviceContainer gnbDev = nr->InstallGnbDevice(gnbs, bwps);
    NetDeviceContainer ueDev = nr->InstallUeDevice(ues, bwps);
    nr->AssignStreams({.gnbDevs = gnbDev, .ueDevs = ueDev});

    for (uint32_t c = 0; c < nCells; ++c)
    {
        w.phys.push_back(NrHelper::GetGnbPhy(gnbDev.Get(c), 0));
        w.phys.back()->TraceConnectWithoutContext("SlotDataStats", MakeCallback(&SlotStats));
    }
    w.ptx.assign(nCells, 0.0);
    w.loadMult.assign(nCells, 1.0);
    w.cellSec.assign(nCells, CellSec{});
    w.utilHist.assign(nCells, {});
    w.violHist.assign(nCells, {});

    auto [remoteHost, remoteAddr] = epc->SetupRemoteHost("100Gb/s", 2500, Seconds(0));
    InternetStackHelper internet;
    internet.Install(ues);
    Ipv4InterfaceContainer ueIp = epc->AssignUeIpv4Address(ueDev);

    // Strongest-received-power attachment at nominal power; 5G-LENA has no handover, so the
    // serving cell stays fixed and a power change acts through SINR and interference only.
    for (uint32_t u = 0; u < ueDev.GetN(); ++u)
    {
        Ptr<MobilityModel> um = ues.Get(u)->GetObject<MobilityModel>();
        double best = -1e300;
        uint32_t bestC = 0;
        for (uint32_t c = 0; c < nCells; ++c)
        {
            Ptr<MobilityModel> gm = gnbs.Get(c)->GetObject<MobilityModel>();
            double rx = w.phys[c]
                            ->GetSpectrumPhy()
                            ->GetSpectrumChannel()
                            ->GetPropagationLossModel()
                            ->CalcRxPower(w.baseTxDbm, gm, um);
            if (rx > best)
            {
                best = rx;
                bestC = c;
            }
        }
        nr->AttachToGnb(ueDev.Get(u), gnbDev.Get(bestC));
        w.ueCell.push_back(bestC);
    }

    Ptr<ExponentialRandomVariable> rateVar = CreateObject<ExponentialRandomVariable>();
    rateVar->SetStream(1001);
    rateVar->SetAttribute("Mean", DoubleValue(1.0));
    uint16_t port = 1234;
    ApplicationContainer apps;
    w.ueSec.assign(ues.GetN(), UeSec{});
    for (uint32_t u = 0; u < ues.GetN(); ++u)
    {
        double bps = rateMbps * 1e6 * (0.5 + 0.5 * rateVar->GetValue());
        w.ueRateBps.push_back(bps);
        w.ueSec[u].offeredBytes = bps / 8.0;
        UdpServerHelper sink(port);
        ApplicationContainer s = sink.Install(ues.Get(u));
        s.Get(0)->TraceConnectWithoutContext("RxWithAddresses", MakeBoundCallback(&UeRx, u));
        apps.Add(s);
        UdpClientHelper client(InetSocketAddress(ueIp.GetAddress(u), port));
        client.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
        client.SetAttribute("PacketSize", UintegerValue(w.pktSize));
        client.SetAttribute("Interval", TimeValue(Seconds(w.pktSize * 8.0 / bps)));
        ApplicationContainer cl = client.Install(remoteHost);
        w.clients.push_back(DynamicCast<UdpClient>(cl.Get(0)));
        apps.Add(cl);
    }
    apps.Start(Seconds(0.1));
    apps.Stop(Seconds(simTime + 1));

    std::string mk = "mkdir -p " + out;
    NS_ABORT_IF(std::system(mk.c_str()) != 0);
    w.series.open(out + "/series.csv");
    w.series << "t,cell,ptx,load,thr,viol,ue,util,energy,delay\n";
    {
        std::ofstream topo(out + "/topology.csv");
        topo << "cell,x,y\n";
        for (uint32_t c = 0; c < nCells; ++c)
        {
            Vector p = gnbs.Get(c)->GetObject<MobilityModel>()->GetPosition();
            topo << c << ',' << p.x << ',' << p.y << '\n';
        }
    }

    Simulator::Schedule(Seconds(1), &Tick);
    for (uint32_t c = 0; c < nCells; ++c)
    {
        double phase = w.period * (1.0 + double(c) / nCells);
        Simulator::Schedule(Seconds(std::round(phase)), &Decide, c);
    }
    Simulator::Stop(Seconds(simTime + 0.5));
    Simulator::Run();
    Simulator::Destroy();

    std::ofstream uf(out + "/units.csv");
    uf << "t0,cell,xapp,cur,prop,mode,applied,p_accept,p_half,p_reject\n";
    for (const Unit& x : w.units)
    {
        uf << x.t0 << ',' << x.cell << ',' << x.xapp << ',' << x.cur << ',' << x.prop << ','
           << x.mode << ',' << x.applied << ',' << w.pAccept << ',' << w.pHalf << ','
           << w.pReject << '\n';
    }
    return 0;
}
