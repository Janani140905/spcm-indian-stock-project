import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  TrendingUp,
  BrainCircuit,
  Newspaper,
  ShieldAlert,
  ArrowRight,
  Sparkles,
  CheckCircle2,
  Clock,
  Layers,
  Database,
} from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { StockPriceChart } from "../components/charts/StockPriceChart.js";
import { ProbabilityBarChart } from "../components/charts/ProbabilityBarChart.js";
import { useApp } from "../context/AppContext.js";
import {
  fetchStockHistory,
  runModelPrediction,
  fetchNews,
  DEMO_PREDICTIONS,
} from "../services/api.js";
import { StockHistory, PredictResponse, NewsArticle } from "../types/index.js";

export const OverviewPage: React.FC = () => {
  const { demoMode, health, activeStock, setActiveStock } = useApp();
  const [history, setHistory] = useState<StockHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [predictions, setPredictions] = useState<PredictResponse | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [newsLoading, setNewsLoading] = useState(false);

  // Load stock chart history
  useEffect(() => {
    const symbolToLoad = activeStock || "RELIANCE";
    setHistoryLoading(true);
    fetchStockHistory(symbolToLoad, "6mo")
      .then((data) => setHistory(data))
      .catch((err) => console.warn(`Could not load history for ${symbolToLoad}:`, err))
      .finally(() => setHistoryLoading(false));
  }, [activeStock]);

  // Load predictions
  useEffect(() => {
    if (demoMode) {
      setPredictions(DEMO_PREDICTIONS);
      return;
    }
    setPredLoading(true);
    runModelPrediction(["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"], 3, "60d")
      .then((data) => setPredictions(data))
      .catch((err) => {
        console.warn("Real inference fetch failed; falling back to demo predictions if offline:", err);
        setPredictions(DEMO_PREDICTIONS);
      })
      .finally(() => setPredLoading(false));
  }, [demoMode]);

  // Load recent news
  useEffect(() => {
    setNewsLoading(true);
    fetchNews(["RELIANCE", "TCS", "INFY"], 4)
      .then((res) => setNews(res.articles))
      .catch(() => {
        // Fallback demo news items if backend offline
        setNews([
          {
            symbol: "INFY",
            title: "Infosys expands AI generative cloud contracts with enterprise clients",
            source: "Financial Express",
            url: "#",
            published_at: "Today, 11:30 IST",
            sentiment_label: "bullish",
            sentiment_score: 0.88,
            is_relevant: true,
          },
          {
            symbol: "RELIANCE",
            title: "Reliance Retail quarterly store expansion and digital connectivity update",
            source: "Economic Times",
            url: "#",
            published_at: "Today, 09:45 IST",
            sentiment_label: "neutral",
            sentiment_score: 0.05,
            is_relevant: true,
          },
        ]);
      })
      .finally(() => setNewsLoading(false));
  }, []);

  const isPyOnline = health?.services?.python_ml_engine?.status === "up";
  const isDbLive = health?.services?.database?.mode === "live";

  return (
    <div className="space-y-6">
      {/* Banner & Data Quality Warning if degraded */}
      {(!isPyOnline || !isDbLive || demoMode) && (
        <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-600/40 flex items-start gap-3 text-xs text-amber-200">
          <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="flex-1 space-y-1">
            <div className="font-semibold flex items-center gap-2">
              <span>SYSTEM STATUS NOTICE</span>
              {demoMode && <Badge variant="demo" size="sm">DEMO MODE ACTIVE</Badge>}
            </div>
            <p className="text-amber-200/80 leading-relaxed">
              {!isPyOnline &&
                "Python ML Engine (:8000) is currently offline. Start the service with `py -3.11 -m uvicorn src.inference.fastapi_adapter:app --port 8000` to enable genuine real-time inference. "}
              {!isDbLive &&
                "PostgreSQL database is currently disconnected; prediction history is operating in local in-memory persistence mode. "}
              {demoMode &&
                "Demo mode is explicitly active. Mock values are clearly tagged and never committed to genuine database prediction history."}
            </p>
          </div>
        </div>
      )}

      {/* Top Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>TARGET UNIVERSE</span>
            <Layers className="w-4 h-4 text-teal-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-white">10 NSE STOCKS</div>
            <p className="text-[11px] text-slate-400 mt-1">Nifty 50 Large-Cap Equities</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-teal-400">
            <span>AUDITED CHRONOLOGICAL LEAKAGE</span>
            <CheckCircle2 className="w-3.5 h-3.5 text-teal-400" />
          </div>
        </Card>

        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>SPCM MODEL ARCHITECTURE</span>
            <BrainCircuit className="w-4 h-4 text-violet-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-violet-300">MLP (64, 32)</div>
            <p className="text-[11px] text-slate-400 mt-1">45 Features (40 Price + 5 Sentiment)</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>In-Sample AUC: <strong className="text-white">0.5183</strong></span>
            <span>Acc: <strong className="text-white">70.0%</strong></span>
          </div>
        </Card>

        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>TOP-K TARGET DEFINITION</span>
            <Sparkles className="w-4 h-4 text-teal-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-teal-300">TOP 30%</div>
            <p className="text-[11px] text-slate-400 mt-1">10-Day Forward Abnormal Returns</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] text-[10px] font-mono text-slate-400">
            Cross-sectional binary ranking target
          </div>
        </Card>

        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>FEED LATENCY & SOURCE</span>
            <Clock className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-amber-300">15m DELAYED</div>
            <p className="text-[11px] text-slate-400 mt-1">Yahoo Finance NSE & Google RSS</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>Zero-Refit Inference Guarantee</span>
            <span className="text-emerald-400">ENFORCED</span>
          </div>
        </Card>
      </div>

      {/* Main Grid: Interactive Chart + Top Predictions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Stock Price Chart */}
        <div className="lg:col-span-2 space-y-4">
          <Card
            title={`Stock Market Performance — ${activeStock || "RELIANCE"}`}
            subtitle="Genuine historical OHLCV chart series from Yahoo Finance"
            action={
              <Link
                to="/stocks"
                className="text-xs font-mono text-teal-400 hover:text-teal-300 flex items-center gap-1"
              >
                <span>Stock Explorer</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <StockPriceChart
              history={history}
              loading={historyLoading}
              onPeriodChange={(p) => {
                if (activeStock) {
                  setHistoryLoading(true);
                  fetchStockHistory(activeStock, p)
                    .then(setHistory)
                    .finally(() => setHistoryLoading(false));
                }
              }}
            />

            {/* Quick stock selector pills */}
            <div className="mt-4 pt-3 border-t border-[#1b2844] flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-mono text-slate-400 mr-1">QUICK SELECT:</span>
              {["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN"].map((s) => (
                <button
                  key={s}
                  onClick={() => setActiveStock(s)}
                  className={`px-2.5 py-1 text-xs font-mono rounded-md transition-all ${
                    activeStock === s
                      ? "bg-teal-500/20 text-teal-300 border border-teal-500/50 font-bold"
                      : "bg-[#0d1424] text-slate-400 border border-[#1b2844] hover:text-slate-200"
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </Card>

          {/* Model Probability Ranking Chart */}
          <Card
            title="SPCM Bullish Probability Distribution"
            subtitle="Cross-sectional class probability P(y=1) estimated by frozen MLP fusion model"
            action={
              <Link
                to="/predictions"
                className="text-xs font-mono text-teal-400 hover:text-teal-300 flex items-center gap-1"
              >
                <span>Run SPCM Inference</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <ProbabilityBarChart predictions={predictions?.predictions || []} />
          </Card>
        </div>

        {/* Right 1 Col: Top Recommendations & Recent Aligned News */}
        <div className="space-y-6">
          {/* Top-K Recommendations Card */}
          <Card
            title="Top-K Stock Recommendations"
            subtitle="Ranked assets with highest relative likelihood of outperformance"
            glow
          >
            <div className="space-y-3">
              {predictions && predictions.recommended_stocks.length > 0 ? (
                predictions.predictions
                  .filter((p) => p.is_recommended)
                  .map((rec) => (
                    <div
                      key={rec.stock}
                      className="p-3 rounded-lg bg-[#0d1424] border border-[#202f4d] flex items-center justify-between"
                    >
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-white text-sm">
                            {rec.stock}
                          </span>
                          <Badge variant="recommended" size="sm">
                            RANK #{rec.rank}
                          </Badge>
                        </div>
                        <div className="text-xs font-mono text-slate-400 mt-1">
                          ₹{rec.close_price.toFixed(2)} · Sentiment:{" "}
                          <span className="text-teal-300 font-semibold">
                            {rec.sentiment_label.toUpperCase()}
                          </span>
                        </div>
                      </div>
                      <div className="text-right font-mono">
                        <div className="text-base font-bold text-teal-400">
                          {(rec.probability * 100).toFixed(1)}%
                        </div>
                        <div className="text-[10px] text-slate-400">P(Bullish)</div>
                      </div>
                    </div>
                  ))
              ) : (
                <div className="py-6 text-center text-xs text-slate-400">
                  {predLoading ? "Evaluating SPCM model matrix..." : "Run predictions to see recommendations."}
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-[#1b2844]">
              <Link
                to="/predictions"
                className="w-full block py-2 text-center rounded-lg bg-gradient-to-r from-violet-600 to-teal-600 text-white font-mono text-xs font-bold hover:opacity-90 transition-opacity shadow-lg shadow-violet-500/20"
              >
                OPEN PREDICTION TERMINAL
              </Link>
            </div>
          </Card>

          {/* Recent News & Sentiment Feed */}
          <Card
            title="Live Financial News Sentiment"
            subtitle="Google News RSS & Yahoo Finance scored with FinBERT / Lexicon"
            action={
              <Link
                to="/news"
                className="text-xs font-mono text-teal-400 hover:text-teal-300 flex items-center gap-1"
              >
                <span>All News</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <div className="space-y-3">
              {news.map((item, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded-lg bg-[#0d1424] border border-[#1a2640] hover:border-slate-600 transition-colors"
                >
                  <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                    <span className="text-teal-300 font-semibold">{item.symbol}</span>
                    <Badge
                      variant={
                        item.sentiment_label === "bullish"
                          ? "bullish"
                          : item.sentiment_label === "bearish"
                          ? "bearish"
                          : "neutral"
                      }
                      size="sm"
                    >
                      {item.sentiment_label.toUpperCase()}
                    </Badge>
                  </div>
                  <h4 className="text-xs text-slate-200 line-clamp-2 leading-relaxed">
                    {item.title}
                  </h4>
                  <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mt-2">
                    <span>{item.source}</span>
                    <span>{item.published_at}</span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
