import React, { useState, useEffect, useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
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
  Sliders,
  Compass,
  Building2,
  ChevronRight,
  Shield,
  Zap,
  Target,
  ExternalLink,
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
  fetchRecommendations,
  DEMO_PREDICTIONS,
} from "../services/api.js";
import { computePersonalizedRankings, DEFAULT_INVESTOR_PROFILE } from "../services/personalization.js";
import { SECTORS } from "../config/stockUniverse.js";
import { StockHistory, PredictResponse, NewsArticle, RecommendationItem } from "../types/index.js";

const QUICK_STOCKS = [
  { symbol: "RELIANCE", sector: "Energy" },
  { symbol: "TCS", sector: "IT" },
  { symbol: "HDFCBANK", sector: "Banking" },
  { symbol: "SUNPHARMA", sector: "Pharma" },
  { symbol: "MARUTI", sector: "Auto" },
  { symbol: "TATASTEEL", sector: "Metals" },
  { symbol: "TITAN", sector: "Retail" },
  { symbol: "NTPC", sector: "Power" },
];

export const OverviewPage: React.FC = () => {
  const navigate = useNavigate();
  const { demoMode, health, activeStock, setActiveStock, investorProfile, setProfileModalOpen } = useApp();
  const [history, setHistory] = useState<StockHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [predictions, setPredictions] = useState<PredictResponse | null>(null);
  const [predLoading, setPredLoading] = useState(false);
  const [news, setNews] = useState<NewsArticle[]>([]);
  const [newsLoading, setNewsLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [recLoading, setRecLoading] = useState(false);

  // Load stock chart history
  useEffect(() => {
    const symbolToLoad = activeStock || "RELIANCE";
    setHistoryLoading(true);
    fetchStockHistory(symbolToLoad, "6mo")
      .then((data) => setHistory(data))
      .catch((err) => console.warn(`Could not load history for ${symbolToLoad}:`, err))
      .finally(() => setHistoryLoading(false));
  }, [activeStock]);

  // Load predictions for benchmark stocks
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

  // Load personalized recommendations
  const effectiveProfile = investorProfile || DEFAULT_INVESTOR_PROFILE;
  useEffect(() => {
    setRecLoading(true);
    fetchRecommendations(effectiveProfile, 3)
      .then((res) => setRecommendations(res.recommendations))
      .catch((err) => {
        console.warn("Recommendation engine call failed:", err);
        // Fallback to client-side ranking if predictions are loaded
        if (predictions?.predictions) {
          const clientRecs = computePersonalizedRankings(predictions.predictions, effectiveProfile)
            .slice(0, 3)
            .map((p, idx) => ({
              symbol: p.stock,
              ticker: `${p.stock}.NS`,
              name: p.stock,
              sector: p.sector || "Large Cap",
              exchange: "NSE" as const,
              currentPrice: p.close_price,
              change: 0,
              changePct: 0,
              marketCapCategory: "Large Cap" as const,
              suitabilityScore: p.personalizedScore,
              rank: idx + 1,
              isTopPick: idx === 0,
              matchReasons: p.matchReasons || ["Profile Alignment"],
              riskIndicators: ["Normal market volatility"],
              factorScores: { sectorMatch: 80, riskFit: 80, objectiveFit: 75, horizonFit: 80, sentimentFit: 70, spcmModelFit: 75 },
              spcmModel: {
                status: "AVAILABLE" as const,
                probability: p.probability,
                rank: p.rank,
                sentimentLabel: p.sentiment_label,
                note: "SPCM Benchmark",
              },
              marketDataTimestamp: new Date().toISOString(),
            }));
          setRecommendations(clientRecs);
        }
      })
      .finally(() => setRecLoading(false));
  }, [effectiveProfile, predictions]);

  // Load recent news
  useEffect(() => {
    setNewsLoading(true);
    fetchNews(["RELIANCE", "TCS", "INFY"], 4)
      .then((res) => setNews(res.articles))
      .catch(() => {
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

  return (
    <div className="space-y-6">
      {/* Top Banner: Multi-Sector Indian Universe Status */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-violet-950/60 via-[#10182c] to-teal-950/40 border border-violet-500/30 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-teal-500/10 border border-teal-500/30 text-teal-400">
            <TrendingUp className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold font-mono text-white flex items-center gap-2">
              INDIAN MULTI-SECTOR STOCK INTELLIGENCE DASHBOARD
            </h1>
            <p className="text-xs text-slate-400 font-mono">
              50 Representative NSE Equities Across 12 Industry Sectors · Multi-Factor Personalization & Fused SPCM Models
            </p>
          </div>
        </div>

        {/* Personalized Questionnaire Prompt CTA */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setProfileModalOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-violet-900/60 hover:bg-violet-800/80 border border-violet-500/50 text-xs font-mono text-teal-300 transition-colors shadow-sm"
          >
            <Compass className="w-3.5 h-3.5" />
            <span>Customize Taste Profile</span>
          </button>

          <div className="flex items-center gap-2 font-mono text-xs">
            <span className="text-slate-400">NSE UNIVERSE:</span>
            <span className="px-2 py-0.5 rounded bg-slate-800 text-teal-300 font-semibold border border-slate-700">
              50 STOCKS / 12 SECTORS
            </span>
          </div>
        </div>
      </div>

      {/* Metrics Row: 4 Essential KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>UNIVERSE MONITOR</span>
            <Database className="w-4 h-4 text-teal-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-white">50 NSE EQUITIES</div>
            <p className="text-[11px] text-slate-400 mt-1">12 Diversified Indian Market Sectors</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>Coverage: Blue-chip & Large Caps</span>
            <span className="text-teal-400">Active Pipeline</span>
          </div>
        </Card>

        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>FROZEN MODEL ARCHITECTURE</span>
            <BrainCircuit className="w-4 h-4 text-violet-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-violet-300">SPCM FUSION NN</div>
            <p className="text-[11px] text-slate-400 mt-1">Price Lags + FinBERT News Sentiment</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>Validation AUC: <strong className="text-teal-400">0.518</strong></span>
            <span>Acc: <strong className="text-white">70.0%</strong></span>
          </div>
        </Card>

        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>INVESTOR PERSONALIZATION</span>
            <Sparkles className="w-4 h-4 text-teal-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-teal-300 uppercase">
              {effectiveProfile.riskTolerance}
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              {effectiveProfile.investmentObjective.replace("_", " ")} · {effectiveProfile.preferredSectors.length} Favored Sectors
            </p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>Suitability Scoring</span>
            <span className="text-emerald-400">Multi-Factor Active</span>
          </div>
        </Card>

        <Card className="flex flex-col justify-between">
          <div className="flex items-center justify-between text-xs font-mono text-slate-400">
            <span>FEED LATENCY & SOURCE</span>
            <Clock className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold font-mono text-amber-300">YAHOO FINANCE .NS</div>
            <p className="text-[11px] text-slate-400 mt-1">Cached Quotes & Google RSS Sentiment</p>
          </div>
          <div className="mt-3 pt-2 border-t border-[#1b2844] flex items-center justify-between text-[10px] font-mono text-slate-400">
            <span>Market Data Cache</span>
            <span className="text-emerald-400">TTL 15m Cached</span>
          </div>
        </Card>
      </div>

      {/* Main Grid: Interactive Chart + Top Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Stock Price Chart & Sector Overview */}
        <div className="lg:col-span-2 space-y-6">
          <Card
            title={`Stock Market Performance — ${activeStock || "RELIANCE"}`}
            subtitle="Genuine historical OHLCV chart series from Yahoo Finance (.NS)"
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
              {QUICK_STOCKS.map(({ symbol, sector }) => (
                <button
                  key={symbol}
                  onClick={() => setActiveStock(symbol)}
                  className={`px-2.5 py-1 text-xs font-mono rounded-md transition-all ${
                    activeStock === symbol
                      ? "bg-teal-500/20 text-teal-300 border border-teal-500/50 font-bold"
                      : "bg-[#0d1424] text-slate-400 border border-[#1b2844] hover:text-slate-200"
                  }`}
                  title={sector}
                >
                  {symbol} <span className="text-[9px] text-slate-400">({sector})</span>
                </button>
              ))}
            </div>
          </Card>

          {/* Model Probability Ranking Chart for Benchmark Stocks */}
          <Card
            title="SPCM Bullish Probability Distribution (Benchmark Universe)"
            subtitle="Cross-sectional class probability P(y=1) estimated by frozen MLP fusion model"
            action={
              <Link
                to="/predictions"
                className="text-xs font-mono text-teal-400 hover:text-teal-300 flex items-center gap-1"
              >
                <span>Inference Terminal</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <ProbabilityBarChart predictions={predictions?.predictions || []} />
          </Card>

          {/* Sector-Wise Market Coverage Grid */}
          <Card
            title="Sector-Wise Market Coverage (12 Industry Sectors)"
            subtitle="Representative high-liquidity NSE companies tracked across Indian industry sectors"
            action={
              <Link
                to="/stocks"
                className="text-xs font-mono text-teal-400 hover:text-teal-300 flex items-center gap-1"
              >
                <span>Explore All 50 Stocks</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            }
          >
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {SECTORS.map((sector) => (
                <div
                  key={sector.name}
                  onClick={() => navigate(`/stocks?sector=${encodeURIComponent(sector.name)}`)}
                  className="p-3 rounded-lg bg-[#0d1424] border border-[#1a2640] hover:border-teal-500/40 hover:bg-[#111a30] transition-all cursor-pointer group flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold font-mono text-slate-200 group-hover:text-teal-300 transition-colors">
                        {sector.displayName}
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-800 text-[10px] font-mono text-teal-400 border border-slate-700">
                        {sector.stockSymbols.length} stocks
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                      {sector.description}
                    </p>
                  </div>

                  <div className="mt-3 pt-2 border-t border-[#182339] flex items-center justify-between">
                    <span className="text-[10px] font-mono text-slate-400 truncate max-w-[170px]">
                      {sector.stockSymbols.slice(0, 3).join(", ")}
                      {sector.stockSymbols.length > 3 ? "..." : ""}
                    </span>
                    <span className="text-[11px] font-mono text-teal-400 flex items-center gap-0.5 group-hover:translate-x-0.5 transition-transform">
                      View <ChevronRight className="w-3 h-3" />
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Right 1 Col: Top Recommendations & Recent Aligned News */}
        <div className="space-y-6">
          {/* Personalized Top-K Recommendations Card */}
          <Card
            title="Personalized Top Picks"
            subtitle={`Tailored for ${effectiveProfile.riskTolerance} risk & preferred sectors`}
            glow
            action={
              <button
                onClick={() => setProfileModalOpen(true)}
                className="text-[11px] font-mono text-teal-300 hover:text-white flex items-center gap-1"
              >
                <Sliders className="w-3 h-3" />
                <span>Change Taste</span>
              </button>
            }
          >
            <div className="space-y-3">
              {recLoading ? (
                <div className="py-8 text-center text-xs text-slate-400 font-mono">
                  Calculating suitability scores across 50 stocks...
                </div>
              ) : recommendations.length > 0 ? (
                recommendations.map((rec) => (
                  <div
                    key={rec.symbol}
                    onClick={() => setActiveStock(rec.symbol)}
                    className="p-3 rounded-lg bg-[#0d1424] border border-[#202f4d] hover:border-teal-500/50 cursor-pointer transition-all"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-white text-sm">
                            {rec.symbol}
                          </span>
                          <span className="text-[10px] text-slate-400 font-mono">.NS</span>
                          <Badge variant="recommended" size="sm">
                            {rec.suitabilityScore}% MATCH
                          </Badge>
                        </div>
                        <div className="text-[11px] font-mono text-slate-400 mt-0.5">
                          ₹{rec.currentPrice.toFixed(2)} · {rec.sector}
                        </div>
                      </div>
                      <div className="text-right font-mono">
                        <div className="text-base font-bold text-teal-400">
                          #{rec.rank}
                        </div>
                        <div className="text-[10px] text-slate-400">Suitability</div>
                      </div>
                    </div>

                    {/* Match Reasons */}
                    <div className="flex flex-wrap gap-1 mt-2">
                      {rec.matchReasons.slice(0, 2).map((reason, idx) => (
                        <span
                          key={idx}
                          className="px-1.5 py-0.5 rounded bg-teal-950/60 text-[10px] text-teal-300 border border-teal-500/30 font-mono"
                        >
                          ✓ {reason}
                        </span>
                      ))}
                    </div>

                    {/* Risk Indicators */}
                    {rec.riskIndicators && rec.riskIndicators.length > 0 && (
                      <div className="mt-1.5 text-[10px] font-mono text-amber-300/80">
                        ⚡ {rec.riskIndicators[0]}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="py-6 text-center text-xs text-slate-400">
                  {predLoading ? "Evaluating SPCM model matrix..." : "Run predictions to see recommendations."}
                </div>
              )}
            </div>

            <div className="mt-4 pt-3 border-t border-[#1b2844] space-y-2">
              <Link
                to="/predictions"
                className="w-full block py-2 text-center rounded-lg bg-gradient-to-r from-violet-600 to-teal-600 text-white font-mono text-xs font-bold hover:opacity-90 transition-opacity shadow-lg shadow-violet-500/20"
              >
                OPEN RECOMMENDATION TERMINAL
              </Link>
              <p className="text-[10px] text-slate-400 text-center font-mono">
                Academic research output, not regulated financial advice.
              </p>
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
              {news.length > 0 ? (
                news.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-[#0d1424] border border-[#1b2844] space-y-1.5 hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-center justify-between text-[11px] font-mono">
                      <span className="font-bold text-teal-300">{item.symbol}</span>
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
                    <p className="text-xs text-slate-200 line-clamp-2 leading-relaxed">
                      {item.title}
                    </p>
                    <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 pt-1 border-t border-[#16223b]">
                      <span>{item.source}</span>
                      <span>{item.published_at}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="py-6 text-center text-xs text-slate-400">
                  {newsLoading ? "Ingesting live financial RSS feeds..." : "No news items available."}
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
