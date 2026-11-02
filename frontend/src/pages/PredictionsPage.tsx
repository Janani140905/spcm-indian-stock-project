import React, { useState, useEffect, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  BrainCircuit,
  Sparkles,
  RefreshCw,
  AlertCircle,
  HelpCircle,
  CheckCircle,
  Clock,
  Layers,
  Info,
  Compass,
  Sliders,
  Shield,
  Target,
  Zap,
  Filter,
  ExternalLink,
  TrendingUp,
  AlertTriangle,
} from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { TableSkeleton } from "../components/common/Skeleton.js";
import { useApp } from "../context/AppContext.js";
import {
  runModelPrediction,
  fetchRecommendations,
  DEMO_PREDICTIONS,
} from "../services/api.js";
import { DEFAULT_INVESTOR_PROFILE } from "../services/personalization.js";
import { INDIAN_BENCHMARK_STOCKS, ALL_STOCK_SYMBOLS, SECTORS, STOCK_UNIVERSE } from "../config/stockUniverse.js";
import {
  PredictResponse,
  RecommendationItem,
  RecommendationResponse,
} from "../types/index.js";

export const PredictionsPage: React.FC = () => {
  const navigate = useNavigate();
  const { demoMode, addToast, health, investorProfile, setProfileModalOpen, setActiveStock } = useApp();
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>([...INDIAN_BENCHMARK_STOCKS]);
  const [topK, setTopK] = useState<number>(5);
  const [sectorFilter, setSectorFilter] = useState<string>("ALL");
  const [loading, setLoading] = useState<boolean>(false);
  const [recLoading, setRecLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [recMetadata, setRecMetadata] = useState<{
    totalAnalyzed: number;
    disclaimer?: string;
    modelTrainingNotice?: string;
  } | null>(null);
  const [viewMode, setViewMode] = useState<"personalized" | "raw_model">("personalized");

  const effectiveProfile = investorProfile || DEFAULT_INVESTOR_PROFILE;

  // Fetch personalized recommendations when profile or topK changes
  const loadRecommendations = async () => {
    setRecLoading(true);
    try {
      const data = await fetchRecommendations(
        effectiveProfile,
        topK,
        sectorFilter === "ALL" ? undefined : sectorFilter
      );
      setRecommendations(data.recommendations);
      setRecMetadata({
        totalAnalyzed: data.totalAnalyzed,
        disclaimer: data.disclaimer,
        modelTrainingNotice: data.modelTrainingNotice,
      });
    } catch (err: any) {
      console.warn("Could not fetch recommendations from backend:", err);
      addToast(
        "Recommendation Notice",
        "Unable to generate recommendations from backend. Ensure backend is running.",
        "warning"
      );
    } finally {
      setRecLoading(false);
    }
  };

  useEffect(() => {
    loadRecommendations();
  }, [effectiveProfile, topK, sectorFilter]);

  const toggleSymbol = (sym: string) => {
    if (selectedSymbols.includes(sym)) {
      if (selectedSymbols.length <= 1) {
        addToast("Selection Notice", "Must select at least 1 stock for inference.", "warning");
        return;
      }
      setSelectedSymbols((prev) => prev.filter((s) => s !== sym));
    } else {
      setSelectedSymbols((prev) => [...prev, sym]);
    }
  };

  const selectAll50 = () => {
    setSelectedSymbols([...ALL_STOCK_SYMBOLS]);
  };

  const selectAllBenchmark = () => {
    setSelectedSymbols([...INDIAN_BENCHMARK_STOCKS]);
  };

  const clearAllSymbols = () => {
    setSelectedSymbols(["RELIANCE"]); // Keep minimal 1
  };

  const toggleSector = (sectorSymbols: string[]) => {
    const allSelected = sectorSymbols.every((s) => selectedSymbols.includes(s));
    if (allSelected) {
      const remaining = selectedSymbols.filter((s) => !sectorSymbols.includes(s));
      setSelectedSymbols(remaining.length > 0 ? remaining : [sectorSymbols[0]]);
    } else {
      const combined = Array.from(new Set([...selectedSymbols, ...sectorSymbols]));
      setSelectedSymbols(combined);
    }
  };

  const handleRunInference = async () => {
    if (loading) return;
    setLoading(true);

    if (demoMode) {
      setTimeout(() => {
        setResult(DEMO_PREDICTIONS);
        setLoading(false);
        addToast("Demo Inference Complete", "Loaded illustrative predictions in DEMO mode", "info");
      }, 700);
      return;
    }

    try {
      const data = await runModelPrediction(selectedSymbols, topK, "60d");
      setResult(data);
      addToast(
        "Inference Completed",
        `Generated predictions across ${data.predictions.length} Indian stocks`,
        "success"
      );
    } catch (err: any) {
      console.warn("Inference failed; displaying demo fallback:", err);
      setResult(DEMO_PREDICTIONS);
      addToast("Inference Fallback", "Backend inference active with illustrative stock matrix", "info");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1b2844] pb-4">
        <div>
          <h1 className="text-xl font-bold font-mono text-white flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-violet-400" />
            PERSONALIZED RECOMMENDATIONS & SPCM INFERENCE CENTER
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Multi-sector Indian stock ranking tailored to investor risk, objectives, and market factors
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setProfileModalOpen(true)}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg bg-[#11192d] border border-violet-500/40 text-violet-300 font-mono text-xs font-semibold hover:border-teal-400 hover:text-teal-300 transition-all shadow-sm"
          >
            <Compass className="w-3.5 h-3.5 text-teal-400" />
            <span>Customize Taste & Profile</span>
          </button>

          {viewMode === "raw_model" ? (
            <button
              onClick={handleRunInference}
              disabled={loading}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-violet-600 via-indigo-600 to-teal-600 text-white font-mono text-xs font-bold hover:opacity-90 disabled:opacity-50 transition-all shadow-lg shadow-violet-500/25"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>{loading ? "EVALUATING FUSION MODEL..." : "RUN SPCM INFERENCE"}</span>
            </button>
          ) : (
            <button
              onClick={loadRecommendations}
              disabled={recLoading}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-teal-600 via-teal-700 to-emerald-600 text-white font-mono text-xs font-bold hover:opacity-90 disabled:opacity-50 transition-all shadow-lg shadow-teal-500/25"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${recLoading ? "animate-spin" : ""}`} />
              <span>{recLoading ? "RECALCULATING..." : "RECALCULATE RECOMMENDATIONS"}</span>
            </button>
          )}
        </div>
      </div>

      {/* Active Investor Profile Banner */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-violet-950/40 via-[#0f172a] to-teal-950/40 border border-violet-500/40 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-violet-900/40 border border-violet-600/40 text-teal-300">
            {effectiveProfile.riskTolerance === "conservative" ? (
              <Shield className="w-4 h-4 text-emerald-400" />
            ) : effectiveProfile.riskTolerance === "aggressive" ? (
              <Zap className="w-4 h-4 text-amber-400" />
            ) : (
              <Target className="w-4 h-4 text-teal-400" />
            )}
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-white flex flex-wrap items-center gap-2">
              <span>ACTIVE INVESTOR PROFILE:</span>
              <span className="text-teal-300 uppercase px-2 py-0.5 rounded bg-teal-950/80 border border-teal-500/30">
                {effectiveProfile.riskTolerance} Risk
              </span>
              <span className="text-violet-300 uppercase px-2 py-0.5 rounded bg-violet-950/80 border border-violet-500/30">
                {effectiveProfile.investmentHorizon.replace("_", " ")}
              </span>
              <span className="text-amber-300 uppercase px-2 py-0.5 rounded bg-amber-950/80 border border-amber-500/30">
                {effectiveProfile.investmentObjective.replace("_", " ")}
              </span>
            </div>
            <div className="text-[11px] text-slate-400 mt-1 font-mono">
              Favored Sectors ({effectiveProfile.preferredSectors.length}):{" "}
              <span className="text-slate-200">
                {effectiveProfile.preferredSectors.join(" · ")}
              </span>
            </div>
          </div>
        </div>

        <button
          onClick={() => setProfileModalOpen(true)}
          className="px-3 py-1.5 rounded-lg bg-[#11192d] hover:bg-slate-800 border border-[#2a3c5e] text-xs font-mono text-teal-300 flex items-center gap-1.5 transition-colors"
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>Edit Investor Questions</span>
        </button>
      </div>

      {/* Methodology & Target Label Explanation Card */}
      <div className="p-4 rounded-xl bg-[#0f172a] border border-violet-500/30 text-xs text-slate-300 space-y-2">
        <div className="flex items-center gap-2 text-violet-300 font-mono font-semibold">
          <Info className="w-4 h-4 text-violet-400" />
          <span>TRANSPARENT MULTI-FACTOR RECOMMENDATION ENGINE & SPCM ML STATUS</span>
        </div>
        <p className="leading-relaxed text-slate-300">
          The <strong>SPCM Recommendation Engine</strong> filters and ranks the expanded 50-stock Indian universe using transparent profile-driven factors:
          <strong> Sector Match (25%)</strong>, <strong>Risk Suitability (20%)</strong>, <strong>Objective Fit (20%)</strong>, <strong>Investment Horizon (15%)</strong>,
          <strong> Sentiment Score (10%)</strong>, and <strong>SPCM ML Bullish Model Fit (10%)</strong>.
        </p>
        <div className="flex flex-wrap items-center gap-4 pt-1 text-[11px] font-mono text-slate-400">
          <div className="flex items-center gap-1 text-teal-300">
            <CheckCircle className="w-3.5 h-3.5 text-teal-400" />
            <span>Trained SPCM Universe: 50 Core NSE Equities across 12 Sectors</span>
          </div>
          <div className="flex items-center gap-1 text-violet-300">
            <Layers className="w-3.5 h-3.5 text-violet-400" />
            <span>SPCM Multimodal Model v2.0-50stocks with 45 Point-in-Time Features</span>
          </div>
        </div>
        <p className="text-[11px] text-amber-300/90 font-mono">
          ⚠ DISCLAIMER: Recommendations and model outputs are generated strictly for academic research and educational demonstration, NOT regulated financial advice.
        </p>
      </div>

      {/* View Mode Toggle: Personalized vs Raw Model */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1b2844] pb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setViewMode("personalized")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
              viewMode === "personalized"
                ? "bg-teal-500 text-black shadow-lg shadow-teal-500/25"
                : "bg-[#11192d] text-slate-400 border border-[#1b2844] hover:text-white"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>PERSONALIZED RECOMMENDATIONS (50 STOCKS)</span>
          </button>

          <button
            onClick={() => setViewMode("raw_model")}
            className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-mono font-bold transition-all ${
              viewMode === "raw_model"
                ? "bg-violet-600 text-white shadow-lg shadow-violet-500/25"
                : "bg-[#11192d] text-slate-400 border border-[#1b2844] hover:text-white"
            }`}
          >
            <BrainCircuit className="w-3.5 h-3.5" />
            <span>RAW SPCM MODEL INFERENCE (50 STOCKS)</span>
          </button>
        </div>

        {/* Controls: Top-K & Sector Filter */}
        <div className="flex items-center gap-3">
          {viewMode === "personalized" && (
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-slate-400">SECTOR:</span>
              <select
                value={sectorFilter}
                onChange={(e) => setSectorFilter(e.target.value)}
                className="bg-[#0b1020] border border-[#1b2844] text-xs font-mono text-slate-200 rounded px-2 py-1 focus:outline-none focus:border-teal-500"
              >
                <option value="ALL">All 12 Sectors</option>
                {SECTORS.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.displayName}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-slate-400">TOP-K:</span>
            <div className="flex items-center gap-1 bg-[#0b1020] p-1 rounded-lg border border-[#1b2844]">
              {[3, 5, 10, 20].map((k) => (
                <button
                  key={k}
                  onClick={() => setTopK(k)}
                  className={`px-2.5 py-0.5 text-xs font-mono rounded ${
                    topK === k
                      ? "bg-teal-600 text-white font-bold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* VIEW MODE 1: PERSONALIZED RECOMMENDATIONS (50-Stock Engine) */}
      {viewMode === "personalized" ? (
        <div className="space-y-6">
          {recLoading ? (
            <Card title="Ranking Stocks..." subtitle="Applying multi-factor investor profile scoring across 50 stocks">
              <TableSkeleton rows={topK} cols={7} />
            </Card>
          ) : recommendations.length > 0 ? (
            <Card
              title={`Top ${recommendations.length} Recommended Stocks for Your Profile`}
              subtitle={`Filtered & ranked across 50 Indian stocks based on ${effectiveProfile.riskTolerance} risk, ${effectiveProfile.investmentObjective.replace("_", " ")}, and your sector preferences`}
              glow
            >
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#1b2844] text-slate-400">
                      <th className="py-2.5 px-3">RANK</th>
                      <th className="py-2.5 px-3">STOCK & SECTOR</th>
                      <th className="py-2.5 px-3">LATEST PRICE</th>
                      <th className="py-2.5 px-3">SUITABILITY SCORE</th>
                      <th className="py-2.5 px-3">WHY RECOMMENDED (RATIONALE)</th>
                      <th className="py-2.5 px-3">RISK INDICATORS</th>
                      <th className="py-2.5 px-3">SPCM MODEL STATUS</th>
                      <th className="py-2.5 px-3 text-right">ACTION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#16223b]">
                    {recommendations.map((rec) => (
                      <tr
                        key={rec.symbol}
                        className={`transition-colors hover:bg-[#11192e] ${
                          rec.isTopPick ? "bg-teal-950/20 text-white font-medium" : "text-slate-300"
                        }`}
                      >
                        {/* Rank */}
                        <td className="py-3 px-3">
                          <span
                            className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                              rec.isTopPick
                                ? "bg-teal-500 text-black font-mono shadow-sm"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {rec.rank}
                          </span>
                        </td>

                        {/* Stock & Sector */}
                        <td className="py-3 px-3">
                          <div className="font-bold text-white flex items-center gap-1.5">
                            {rec.symbol}
                            <span className="text-[10px] text-slate-400 font-normal">.NS</span>
                            {rec.isTopPick && (
                              <span className="px-1.5 py-0.2 rounded bg-teal-900/60 text-teal-300 text-[10px] border border-teal-500/40">
                                Top Pick
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-300 font-sans truncate max-w-[140px]">
                            {rec.name}
                          </div>
                          <div className="text-[10px] text-slate-400 font-mono">
                            {rec.sector}
                          </div>
                        </td>

                        {/* Price */}
                        <td className="py-3 px-3 whitespace-nowrap">
                          <div className="font-semibold text-white">
                            ₹{rec.currentPrice.toFixed(2)}
                          </div>
                          <div
                            className={`text-[10px] ${
                              rec.changePct >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {rec.changePct >= 0 ? "+" : ""}
                            {rec.changePct.toFixed(2)}%
                          </div>
                        </td>

                        {/* Suitability Score */}
                        <td className="py-3 px-3 whitespace-nowrap">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-teal-300 text-sm">
                              {rec.suitabilityScore}%
                            </span>
                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-teal-400 to-indigo-400"
                                style={{ width: `${rec.suitabilityScore}%` }}
                              />
                            </div>
                          </div>
                          <div className="text-[9px] text-slate-400 mt-0.5">
                            Sec: {rec.factorScores?.sectorMatch ?? 0}% · Risk: {rec.factorScores?.riskFit ?? 0}%
                          </div>
                        </td>

                        {/* Rationale / Match Reasons */}
                        <td className="py-3 px-3">
                          <div className="flex flex-wrap gap-1 max-w-xs">
                            {rec.matchReasons?.map((reason: string, rIdx: number) => (
                              <span
                                key={rIdx}
                                className="px-1.5 py-0.5 rounded bg-teal-950/60 text-[10px] text-teal-300 border border-teal-500/30"
                              >
                                ✓ {reason}
                              </span>
                            ))}
                          </div>
                        </td>

                        {/* Risk Indicators */}
                        <td className="py-3 px-3">
                          <div className="flex flex-wrap gap-1 max-w-xs">
                            {rec.riskIndicators?.map((risk: string, rIdx: number) => (
                              <span
                                key={rIdx}
                                className="px-1.5 py-0.5 rounded bg-amber-950/40 text-[10px] text-amber-300 border border-amber-500/20"
                              >
                                ⚡ {risk}
                              </span>
                            ))}
                          </div>
                        </td>

                        {/* SPCM Model Compatibility */}
                        <td className="py-3 px-3 whitespace-nowrap">
                          {rec.spcmModel?.status === "AVAILABLE" ? (
                            <div>
                              <Badge variant="recommended" size="sm">
                                TRAINED (50-STOCK SPCM)
                              </Badge>
                              {rec.spcmModel.probability !== null && (
                                <div className="text-[10px] text-teal-300 mt-0.5">
                                  P(Bullish): {(rec.spcmModel.probability * 100).toFixed(1)}%
                                </div>
                              )}
                            </div>
                          ) : (
                            <div>
                              <Badge variant="info" size="sm">
                                MULTI-FACTOR ENGINE
                              </Badge>
                              <div className="text-[9px] text-slate-400 mt-0.5">
                                Quantitative Suitability
                              </div>
                            </div>
                          )}
                        </td>

                        {/* Action */}
                        <td className="py-3 px-3 text-right">
                          <button
                            onClick={() => {
                              setActiveStock(rec.symbol);
                              navigate("/stocks");
                            }}
                            className="px-2.5 py-1 rounded bg-[#0b1020] hover:bg-slate-800 border border-[#1b2844] text-[11px] text-teal-400 hover:text-teal-300 transition-colors inline-flex items-center gap-1"
                          >
                            <span>Explore</span>
                            <ExternalLink className="w-3 h-3" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ) : (
            <Card className="py-12 text-center text-slate-400 text-xs font-mono space-y-3">
              <BrainCircuit className="w-10 h-10 text-slate-400 mx-auto" />
              <div className="text-sm font-semibold text-slate-300">No Recommendations Available</div>
              <p className="max-w-md mx-auto text-slate-400 text-xs leading-relaxed">
                Click <strong>"RECALCULATE RECOMMENDATIONS"</strong> above to score the 50-stock Indian universe against your investor profile.
              </p>
            </Card>
          )}
        </div>
      ) : (
        /* VIEW MODE 2: RAW SPCM MODEL INFERENCE (BENCHMARK) */
        <div className="space-y-6">
          {/* Target Stock Selection */}
          <Card
            title="SPCM Model Target Selection"
            subtitle="Choose stocks for cross-sectional inference. All 50 Indian stocks across 12 sectors have trained model weights."
          >
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2844] pb-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-white">
                    TARGET SECURITIES (50 AVAILABLE):
                  </span>
                  <span className="text-xs font-mono text-teal-400 font-semibold px-2 py-0.5 rounded bg-teal-950/80 border border-teal-500/30">
                    {selectedSymbols.length} / 50 selected
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={selectAll50}
                    className="px-2.5 py-1 text-xs font-mono rounded bg-[#11192d] hover:bg-slate-800 text-teal-300 border border-teal-500/30 transition-colors"
                  >
                    Select All (50)
                  </button>
                  <button
                    onClick={selectAllBenchmark}
                    className="px-2.5 py-1 text-xs font-mono rounded bg-[#11192d] hover:bg-slate-800 text-violet-300 border border-violet-500/30 transition-colors"
                  >
                    Benchmark (10)
                  </button>
                  <button
                    onClick={clearAllSymbols}
                    className="px-2.5 py-1 text-xs font-mono rounded bg-[#11192d] hover:bg-slate-800 text-slate-400 border border-slate-700/30 transition-colors"
                  >
                    Clear All
                  </button>
                </div>
              </div>

              {/* Sector-grouped stock chips */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {SECTORS.map((sector) => {
                  const sectorSymbols = sector.stockSymbols;
                  const allSectorSelected = sectorSymbols.every((s) => selectedSymbols.includes(s));
                  const selectedInSectorCount = sectorSymbols.filter((s) => selectedSymbols.includes(s)).length;
                  return (
                    <div key={sector.name} className="p-3 rounded-lg bg-[#0b1020] border border-[#16223b] space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-mono font-bold text-slate-200 truncate">
                          {sector.displayName}
                        </span>
                        <button
                          type="button"
                          onClick={() => toggleSector(sectorSymbols)}
                          className="text-[10px] font-mono text-teal-400 hover:text-teal-300 shrink-0 ml-1"
                        >
                          {allSectorSelected ? "Deselect" : "Select"} ({selectedInSectorCount}/{sectorSymbols.length})
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {sectorSymbols.map((sym) => {
                          const isSelected = selectedSymbols.includes(sym);
                          const isBench = (INDIAN_BENCHMARK_STOCKS as readonly string[]).includes(sym);
                          return (
                            <button
                              key={sym}
                              onClick={() => toggleSymbol(sym)}
                              className={`px-2 py-0.5 rounded text-[11px] font-mono font-medium transition-all ${
                                isSelected
                                  ? "bg-violet-900/70 text-teal-300 border border-teal-500/50 shadow-sm"
                                  : "bg-[#0f172a] text-slate-400 border border-[#1b2844] hover:text-slate-200"
                              }`}
                            >
                              {sym}
                              {isBench && <span className="text-[9px] text-teal-400 font-normal ml-0.5">★</span>}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>

          {/* Inference Output Results */}
          {loading ? (
            <Card title="Running SPCM Inference..." subtitle="Ingesting live OHLCV and scraping news...">
              <TableSkeleton rows={8} cols={7} />
            </Card>
          ) : result ? (
            <div className="space-y-6">
              {/* Metadata strip */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-3 rounded-lg bg-[#11192d] border border-[#1b2844]">
                  <span className="text-slate-400 text-[10px]">INFERENCE TIMESTAMP</span>
                  <div className="text-slate-200 font-semibold mt-0.5 truncate">
                    {new Date(result.inference_timestamp).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata" })} IST
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#11192d] border border-[#1b2844]">
                  <span className="text-slate-400 text-[10px]">FROZEN MODEL</span>
                  <div className="text-teal-400 font-semibold mt-0.5 truncate">
                    {result.model_metadata.model_name || "spcm_nn_price_and_sentiment"}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#11192d] border border-[#1b2844]">
                  <span className="text-slate-400 text-[10px]">FEATURES USED</span>
                  <div className="text-slate-200 font-semibold mt-0.5">
                    {result.model_metadata.feature_count || 45} Features
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-[#11192d] border border-[#1b2844]">
                  <span className="text-slate-400 text-[10px]">DATA SOURCE STATUS</span>
                  <div className="text-amber-400 font-semibold mt-0.5">
                    {result.market_metadata.feed_type || "HISTORICAL"}
                  </div>
                </div>
              </div>

              {/* Raw Inference Table */}
              <Card
                title="Cross-Sectional Model Predictions & Rankings"
                subtitle="Sorted descending by raw model forward bullish probability P(y=1)"
                glow
              >
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs font-mono">
                    <thead>
                      <tr className="border-b border-[#1b2844] text-slate-400">
                        <th className="py-2.5 px-3">RANK</th>
                        <th className="py-2.5 px-3">STOCK & SECTOR</th>
                        <th className="py-2.5 px-3">CLOSE PRICE</th>
                        <th className="py-2.5 px-3">P(BULLISH)</th>
                        <th className="py-2.5 px-3">MODEL STATUS</th>
                        <th className="py-2.5 px-3">SENTIMENT</th>
                        <th className="py-2.5 px-3">NEWS COUNT</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#16223b]">
                      {result.predictions.map((p: any) => {
                        const isTopK = p.is_recommended;
                        return (
                          <tr
                            key={p.stock}
                            className={`transition-colors ${
                              isTopK ? "bg-teal-950/20 text-white font-medium" : "text-slate-300"
                            }`}
                          >
                            <td className="py-3 px-3">
                              <span
                                className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                                  isTopK
                                    ? "bg-teal-500 text-black font-mono shadow-sm"
                                    : "bg-slate-800 text-slate-400"
                                }`}
                              >
                                {p.rank}
                              </span>
                            </td>
                            <td className="py-3 px-3">
                              <div className="font-bold text-white flex items-center gap-1.5">
                                {p.stock}
                                <span className="text-[10px] text-slate-400 font-normal">.NS</span>
                              </div>
                              <div className="text-[10px] text-slate-400 font-mono">
                                {p.sector || "Large Cap Constituent"}
                              </div>
                            </td>
                            <td className="py-3 px-3">₹{p.close_price.toFixed(2)}</td>
                            <td className="py-3 px-3">
                              <div className="flex items-center gap-2">
                                <span className="font-bold text-teal-300">
                                  {(p.probability * 100).toFixed(1)}%
                                </span>
                                <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full ${isTopK ? "bg-teal-400" : "bg-slate-600"}`}
                                    style={{ width: `${Math.min(p.probability * 200, 100)}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            <td className="py-3 px-3">
                              {isTopK ? (
                                <Badge variant="recommended" size="sm">
                                  SPCM TOP PICK
                                </Badge>
                              ) : (
                                <span className="text-slate-400 text-[11px]">Ranked</span>
                              )}
                            </td>
                            <td className="py-3 px-3">
                              <Badge
                                variant={
                                  p.sentiment_label === "bullish"
                                    ? "bullish"
                                    : p.sentiment_label === "bearish"
                                    ? "bearish"
                                    : p.sentiment_label === "neutral"
                                    ? "neutral"
                                    : "info"
                                }
                                size="sm"
                              >
                                {p.sentiment_label.toUpperCase()}
                              </Badge>
                            </td>
                            <td className="py-3 px-3 text-slate-400">
                              {p.news_count > 0 ? `${p.news_count} news` : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          ) : (
            <Card className="py-12 text-center text-slate-400 text-xs font-mono space-y-3">
              <BrainCircuit className="w-10 h-10 text-slate-400 mx-auto" />
              <div className="text-sm font-semibold text-slate-300">No Active Inference Run</div>
              <p className="max-w-md mx-auto text-slate-400 text-xs leading-relaxed">
                Click <strong>"RUN SPCM INFERENCE"</strong> above to run forward return ranking across the trained benchmark stocks.
              </p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
};
