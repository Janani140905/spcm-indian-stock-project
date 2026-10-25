import React, { useState } from "react";
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
} from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { TableSkeleton } from "../components/common/Skeleton.js";
import { useApp } from "../context/AppContext.js";
import { runModelPrediction, DEMO_PREDICTIONS } from "../services/api.js";
import { PredictResponse } from "../types/index.js";

const ALL_SYMBOLS = [
  "RELIANCE",
  "TCS",
  "HDFCBANK",
  "INFY",
  "ICICIBANK",
  "HINDUNILVR",
  "ITC",
  "SBIN",
  "BHARTIARTL",
  "KOTAKBANK",
];

export const PredictionsPage: React.FC = () => {
  const { demoMode, addToast, health } = useApp();
  const [selectedSymbols, setSelectedSymbols] = useState<string[]>(ALL_SYMBOLS);
  const [topK, setTopK] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictResponse | null>(null);

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

  const handleRunInference = async () => {
    if (loading) return; // Prevent duplicate requests
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
        `Generated genuine predictions across ${data.predictions.length} Indian stocks`,
        "success"
      );
    } catch (err: any) {
      console.warn("Inference failed:", err);
      addToast(
        "Inference Failed",
        err.message || "Failed to execute inference pipeline. Ensure Python service is running.",
        "error"
      );
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
            ML PREDICTION COMMAND CENTER
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Multimodal fusion inference fusing market OHLCV lags with point-in-time financial news sentiment
          </p>
        </div>

        <button
          onClick={handleRunInference}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-violet-600 via-indigo-600 to-teal-600 text-white font-mono text-xs font-bold hover:opacity-90 disabled:opacity-50 transition-all shadow-lg shadow-violet-500/25"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
          <span>{loading ? "EVALUATING FUSION MODEL..." : "RUN SPCM INFERENCE"}</span>
        </button>
      </div>

      {/* Methodology & Target Label Explanation Card */}
      <div className="p-4 rounded-xl bg-[#0f172a] border border-violet-500/30 text-xs text-slate-300 space-y-2">
        <div className="flex items-center gap-2 text-violet-300 font-mono font-semibold">
          <Info className="w-4 h-4 text-violet-400" />
          <span>TARGET LABEL & PROBABILITY METHODOLOGY</span>
        </div>
        <p className="leading-relaxed text-slate-300">
          The SPCM model predicts binary membership in the <strong>Top 30% of cross-sectional 10-day forward returns</strong> (<code className="text-teal-300">fwd_ret</code>).
          The estimated probability <code className="text-violet-300">P(label=1)</code> quantifies the asset's relative likelihood of outperforming the 10-stock universe over the next 10 trading sessions.
        </p>
        <p className="text-[11px] text-amber-300/90 font-mono">
          ⚠ DISCLAIMER: This probability is an academic research prediction regarding cross-sectional rank, NOT a guaranteed price increase, guaranteed profit, or direct financial advice.
        </p>
      </div>

      {/* Control Panel: Stock Selection & Top-K */}
      <Card title="Inference Configuration" subtitle="Configure target universe subset and recommendation budget">
        <div className="space-y-4">
          <div>
            <label className="text-xs font-mono text-slate-400 block mb-2">
              SELECT TARGET SECURITIES ({selectedSymbols.length} / 10 selected):
            </label>
            <div className="flex flex-wrap gap-2">
              {ALL_SYMBOLS.map((sym) => {
                const selected = selectedSymbols.includes(sym);
                return (
                  <button
                    key={sym}
                    onClick={() => toggleSymbol(sym)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold transition-all ${
                      selected
                        ? "bg-violet-950 text-teal-300 border border-teal-500/50 shadow-sm"
                        : "bg-[#0b1020] text-slate-400 border border-[#1b2844] hover:text-slate-200"
                    }`}
                  >
                    {sym}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-6 pt-3 border-t border-[#1b2844]">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-slate-400">RECOMMENDATION COUNT (TOP-K):</span>
              <div className="flex items-center gap-1 bg-[#0b1020] p-1 rounded-lg border border-[#1b2844]">
                {[1, 2, 3, 5].map((k) => (
                  <button
                    key={k}
                    onClick={() => setTopK(k)}
                    className={`px-3 py-1 text-xs font-mono rounded ${
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

            <div className="text-xs font-mono text-slate-400">
              WARM-UP WINDOW: <span className="text-slate-200 font-bold">60 DAYS</span> (for 1, 5, 10, 30-day technical lags)
            </div>
          </div>
        </div>
      </Card>

      {/* Prediction Output Results */}
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

          {/* Predictions Ranked Table */}
          <Card
            title="Cross-Sectional Model Predictions & Rankings"
            subtitle="Sorted descending by forward bullish probability P(y=1)"
            glow
          >
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#1b2844] text-slate-400">
                    <th className="py-2.5 px-3">RANK</th>
                    <th className="py-2.5 px-3">STOCK</th>
                    <th className="py-2.5 px-3">CLOSE PRICE</th>
                    <th className="py-2.5 px-3">P(BULLISH)</th>
                    <th className="py-2.5 px-3">STATUS</th>
                    <th className="py-2.5 px-3">SENTIMENT</th>
                    <th className="py-2.5 px-3">NEWS HEADLINES</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#16223b]">
                  {result.predictions.map((p) => {
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
                        <td className="py-3 px-3 font-bold text-white">
                          {p.stock}
                          <span className="text-[10px] text-slate-400 font-normal">.NS</span>
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
                              RECOMMENDED
                            </Badge>
                          ) : (
                            <span className="text-slate-400 text-[11px]">—</span>
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
                          {p.news_count > 0 ? `${p.news_count} articles` : "No articles"}
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
            Click <strong>"RUN SPCM INFERENCE"</strong> above to fetch the latest market quotes and news sentiment, fuse them into the 45-feature multimodal matrix, and predict cross-sectional rankings.
          </p>
        </Card>
      )}
    </div>
  );
};
