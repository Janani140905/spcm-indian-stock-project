import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Search,
  Star,
  TrendingUp,
  Layers,
  ArrowUpDown,
  ExternalLink,
  ShieldCheck,
  Check,
  Activity,
  Compass,
  AlertCircle,
  Clock,
  Sparkles,
} from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { StockPriceChart } from "../components/charts/StockPriceChart.js";
import { TableSkeleton } from "../components/common/Skeleton.js";
import { useApp } from "../context/AppContext.js";
import {
  fetchStocks,
  fetchSectors,
  fetchStockQuote,
  fetchStockHistory,
  fetchWatchlist,
  addToWatchlist,
  removeFromWatchlist,
} from "../services/api.js";
import { Stock, StockQuote, StockHistory, WatchlistItem } from "../types/index.js";
import { SECTORS } from "../config/stockUniverse.js";

export const StockExplorerPage: React.FC = () => {
  const { activeStock, setActiveStock, addToast } = useApp();
  const [searchParams, setSearchParams] = useSearchParams();
  const initialSector = searchParams.get("sector") || "ALL";
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sectorFilter, setSectorFilter] = useState(initialSector);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [selectedQuote, setSelectedQuote] = useState<StockQuote | null>(null);
  const [selectedStockHistory, setSelectedStockHistory] = useState<StockHistory | null>(null);
  const [quoteLoading, setQuoteLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Sync with URL query parameter
  useEffect(() => {
    const secParam = searchParams.get("sector");
    if (secParam) {
      setSectorFilter(secParam);
    }
  }, [searchParams]);

  // Load stocks catalog and watchlist
  useEffect(() => {
    setLoading(true);
    Promise.all([fetchStocks(), fetchWatchlist()])
      .then(([stockList, wlItems]) => {
        setStocks(stockList);
        setWatchlist(wlItems);
      })
      .catch((err) => console.warn("Failed to fetch stocks or watchlist:", err))
      .finally(() => setLoading(false));
  }, []);

  // Load selected stock quote and chart history
  useEffect(() => {
    if (!activeStock) return;
    setQuoteLoading(true);
    setHistoryLoading(true);

    fetchStockQuote(activeStock)
      .then(setSelectedQuote)
      .catch((err) => console.warn(`Failed to fetch quote for ${activeStock}:`, err))
      .finally(() => setQuoteLoading(false));

    fetchStockHistory(activeStock, "6mo")
      .then(setSelectedStockHistory)
      .catch((err) => console.warn(`Failed to fetch history for ${activeStock}:`, err))
      .finally(() => setHistoryLoading(false));
  }, [activeStock]);

  const filteredStocks = stocks.filter((s) => {
    const matchSearch =
      s.symbol.toLowerCase().includes(search.toLowerCase()) ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.sector.toLowerCase().includes(search.toLowerCase());
    const matchSector = sectorFilter === "ALL" || s.sector === sectorFilter;
    return matchSearch && matchSector;
  });

  const isWatchlisted = (symbol: string) => watchlist.some((w) => w.symbol === symbol);

  const toggleWatchlist = async (symbol: string) => {
    const cleanSym = symbol.toUpperCase().replace(".NS", "");
    try {
      if (isWatchlisted(cleanSym)) {
        await removeFromWatchlist(cleanSym);
        setWatchlist((prev) => prev.filter((w) => w.symbol !== cleanSym));
        addToast("Watchlist Updated", `Removed ${cleanSym} from watchlist`, "info");
      } else {
        const item = await addToWatchlist({ symbol: cleanSym });
        setWatchlist((prev) => [...prev, item]);
        addToast("Watchlist Updated", `Added ${cleanSym} to watchlist`, "success");
      }
    } catch (err: any) {
      addToast("Watchlist Error", err.message || "Failed to update watchlist", "error");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1b2844] pb-4">
        <div>
          <h1 className="text-xl font-bold font-mono text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-teal-400" />
            EXPANDED MULTI-SECTOR INDIAN STOCK EXPLORER
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            50 High-Liquidity NSE Equities across 12 Diversified Industry Sectors with Genuine Market Data
          </p>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search 50 stocks by symbol, company, or sector..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 pr-3 py-1.5 rounded-lg bg-[#11192d] border border-[#1f2d48] text-xs text-slate-100 placeholder:text-slate-400 focus:outline-none focus:border-teal-500/50 w-72 font-sans"
          />
        </div>
      </div>

      {/* Sector Filter Tabs (12 Sectors + ALL) */}
      <div className="flex items-center gap-2 overflow-x-auto pb-2 scrollbar-thin">
        <button
          onClick={() => setSectorFilter("ALL")}
          className={`px-3 py-1.5 rounded-lg text-xs font-mono font-semibold whitespace-nowrap transition-all ${
            sectorFilter === "ALL"
              ? "bg-teal-500 text-black shadow-md shadow-teal-500/20"
              : "bg-[#11192d] text-slate-400 border border-[#1b2844] hover:text-white"
          }`}
        >
          ALL SECTORS ({stocks.length})
        </button>

        {SECTORS.map((sec) => {
          const isSelected = sectorFilter === sec.name;
          const count = stocks.filter((s) => s.sector === sec.name).length;
          return (
            <button
              key={sec.name}
              onClick={() => setSectorFilter(sec.name)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono font-medium whitespace-nowrap transition-all ${
                isSelected
                  ? "bg-teal-500 text-black font-bold shadow-md shadow-teal-500/20"
                  : "bg-[#11192d] text-slate-400 border border-[#1b2844] hover:text-slate-200"
              }`}
            >
              {sec.displayName} ({count})
            </button>
          );
        })}
      </div>

      {/* 2-Column Layout: Stock Table + Detailed Analysis Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Stock Table (7 cols) */}
        <div className="lg:col-span-7">
          <Card
            title={`Constituent Equities (${filteredStocks.length} Results)`}
            subtitle="Click any row to load verified historical OHLCV candles and live technicals"
          >
            {loading ? (
              <TableSkeleton rows={12} cols={5} />
            ) : (
              <div className="overflow-x-auto max-h-[640px] overflow-y-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="sticky top-0 bg-[#0e1628] border-b border-[#1b2844] text-slate-400 z-10">
                    <tr>
                      <th className="py-2.5 px-3">SYMBOL</th>
                      <th className="py-2.5 px-3">COMPANY NAME</th>
                      <th className="py-2.5 px-3">SECTOR</th>
                      <th className="py-2.5 px-3">SPCM MODEL</th>
                      <th className="py-2.5 px-3 text-right">WATCHLIST</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#16223b]">
                    {filteredStocks.map((stock) => {
                      const selected = activeStock === stock.symbol;
                      const inWatchlist = isWatchlisted(stock.symbol);
                      return (
                        <tr
                          key={stock.symbol}
                          onClick={() => setActiveStock(stock.symbol)}
                          className={`cursor-pointer transition-colors ${
                            selected
                              ? "bg-violet-950/40 text-teal-300 font-semibold"
                              : "hover:bg-[#141e35] text-slate-300"
                          }`}
                        >
                          <td className="py-3 px-3">
                            <div className="flex items-center gap-1.5">
                              <span className="font-bold text-white">{stock.symbol}</span>
                              <span className="text-[10px] text-slate-400">.NS</span>
                            </div>
                          </td>
                          <td className="py-3 px-3 text-slate-200 max-w-[180px] truncate">
                            {stock.name}
                          </td>
                          <td className="py-3 px-3 text-slate-400 text-[11px]">
                            <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800">
                              {stock.sector}
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            {stock.inTrainingUniverse ? (
                              <span className="px-2 py-0.5 rounded bg-teal-950/80 text-teal-300 text-[10px] border border-teal-500/40 font-mono">
                                Benchmark Trained
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px] border border-slate-700 font-mono">
                                Multi-Factor Scored
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-3 text-right" onClick={(e) => e.stopPropagation()}>
                            <button
                              onClick={() => toggleWatchlist(stock.symbol)}
                              className={`p-1.5 rounded transition-colors ${
                                inWatchlist
                                  ? "text-amber-400 hover:text-amber-300"
                                  : "text-slate-400 hover:text-slate-300"
                              }`}
                              title={inWatchlist ? "Remove from watchlist" : "Add to watchlist"}
                            >
                              {inWatchlist ? (
                                <Star className="w-4 h-4 fill-amber-400" />
                              ) : (
                                <Star className="w-4 h-4" />
                              )}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>

        {/* Right Column: Active Stock Details & Recharts Series (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <Card
            title={`Security Deep-Dive — ${activeStock || "Select Stock"}`}
            subtitle={
              selectedQuote
                ? `${selectedQuote.name} · ${selectedQuote.sector}`
                : "Verified historical prices and exchange technicals"
            }
            glow
          >
            {activeStock ? (
              <div className="space-y-4">
                {/* Real Market Price Strip */}
                {selectedQuote && (
                  <div className="p-3.5 rounded-xl bg-[#0b1020] border border-[#1b2844] space-y-3">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="text-2xl font-bold font-mono text-white">
                          ₹{selectedQuote.currentPrice.toFixed(2)}
                        </div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span
                            className={`text-xs font-mono font-bold ${
                              selectedQuote.change >= 0 ? "text-emerald-400" : "text-rose-400"
                            }`}
                          >
                            {selectedQuote.change >= 0 ? "+" : ""}
                            {selectedQuote.change.toFixed(2)} ({selectedQuote.changePct.toFixed(2)}%)
                          </span>
                          <span className="text-[10px] text-slate-400 font-mono">vs Prev Close</span>
                        </div>
                      </div>

                      <div className="text-right">
                        <span
                          className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                            selectedQuote.feedStatus === "DELAYED_15M"
                              ? "bg-amber-950/60 border-amber-500/40 text-amber-300"
                              : "bg-slate-800 text-slate-300 border-slate-700"
                          }`}
                        >
                          {selectedQuote.feedStatus}
                        </span>
                        <div className="text-[10px] text-slate-400 font-mono mt-1">
                          {selectedQuote.marketTimeIST}
                        </div>
                      </div>
                    </div>

                    {/* Technical KPIs: Volatility, Momentum, 52w Range */}
                    <div className="grid grid-cols-3 gap-2 pt-2 border-t border-[#16223b] text-center font-mono">
                      <div className="p-2 rounded bg-[#0f172a] border border-[#1b2844]">
                        <span className="text-[9px] text-slate-400 block">30D VOLATILITY</span>
                        <span className="text-xs font-bold text-teal-300">
                          {selectedQuote.technicalIndicators.volatility30d !== null
                            ? `${selectedQuote.technicalIndicators.volatility30d}%`
                            : "—"}
                        </span>
                      </div>

                      <div className="p-2 rounded bg-[#0f172a] border border-[#1b2844]">
                        <span className="text-[9px] text-slate-400 block">10D MOMENTUM</span>
                        <span
                          className={`text-xs font-bold ${
                            (selectedQuote.technicalIndicators.momentum10d ?? 0) >= 0
                              ? "text-emerald-400"
                              : "text-rose-400"
                          }`}
                        >
                          {selectedQuote.technicalIndicators.momentum10d !== null
                            ? `${selectedQuote.technicalIndicators.momentum10d >= 0 ? "+" : ""}${selectedQuote.technicalIndicators.momentum10d}%`
                            : "—"}
                        </span>
                      </div>

                      <div className="p-2 rounded bg-[#0f172a] border border-[#1b2844]">
                        <span className="text-[9px] text-slate-400 block">14D RSI</span>
                        <span className="text-xs font-bold text-violet-300">
                          {selectedQuote.technicalIndicators.rsi14 !== null
                            ? selectedQuote.technicalIndicators.rsi14
                            : "—"}
                        </span>
                      </div>
                    </div>

                    {/* SPCM Model Compatibility Notice */}
                    <div className="p-2.5 rounded-lg bg-[#0e1628] border border-[#233554] text-[11px] font-mono text-slate-300">
                      {selectedQuote.inTrainingUniverse ? (
                        <div className="flex items-center gap-1.5 text-teal-300 font-semibold">
                          <Check className="w-3.5 h-3.5 text-teal-400" />
                          <span>SPCM Multimodal Model: Supported in Training Benchmark</span>
                        </div>
                      ) : (
                        <div className="flex items-start gap-1.5 text-slate-400">
                          <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                          <span>
                            SPCM Model Compatibility: Evaluated via multi-factor quantitative engine. Full cross-sectional SPCM retraining pipeline required for binary forward probability.
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Price Chart */}
                <StockPriceChart
                  history={selectedStockHistory}
                  loading={historyLoading}
                  onPeriodChange={(p) => {
                    setHistoryLoading(true);
                    fetchStockHistory(activeStock, p)
                      .then(setSelectedStockHistory)
                      .finally(() => setHistoryLoading(false));
                  }}
                />

                {/* Actions */}
                <div className="pt-2 flex items-center justify-between">
                  <button
                    onClick={() => activeStock && toggleWatchlist(activeStock)}
                    className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#141f36] border border-[#2a3c5e] text-xs font-mono text-slate-200 hover:border-teal-500/50"
                  >
                    <Star
                      className={`w-3.5 h-3.5 ${
                        isWatchlisted(activeStock) ? "fill-amber-400 text-amber-400" : ""
                      }`}
                    />
                    <span>
                      {isWatchlisted(activeStock) ? "On Watchlist" : "Add to Watchlist"}
                    </span>
                  </button>

                  <a
                    href={`https://www.nseindia.com/get-quotes/equity?symbol=${activeStock}`}
                    target="_blank"
                    rel="noreferrer"
                    className="flex items-center gap-1.5 text-xs font-mono text-teal-400 hover:underline"
                  >
                    <span>View on NSE Official</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center text-xs text-slate-400 font-mono">
                Select a stock from the table to inspect details.
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};
