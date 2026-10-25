import React, { useState, useEffect } from "react";
import {
  Search,
  Star,
  StarOff,
  TrendingUp,
  Layers,
  ArrowUpDown,
  ExternalLink,
  ShieldCheck,
  Check,
} from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { StockPriceChart } from "../components/charts/StockPriceChart.js";
import { TableSkeleton } from "../components/common/Skeleton.js";
import { useApp } from "../context/AppContext.js";
import {
  fetchStocks,
  fetchStockHistory,
  fetchWatchlist,
  addToWatchlist,
  removeFromWatchlist,
} from "../services/api.js";
import { Stock, StockHistory, WatchlistItem } from "../types/index.js";

export const StockExplorerPage: React.FC = () => {
  const { activeStock, setActiveStock, addToast } = useApp();
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [sectorFilter, setSectorFilter] = useState("ALL");
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [selectedStockHistory, setSelectedStockHistory] = useState<StockHistory | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);

  // Load stocks and watchlist
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

  // Load selected stock history
  useEffect(() => {
    if (!activeStock) return;
    setHistoryLoading(true);
    fetchStockHistory(activeStock, "6mo")
      .then(setSelectedStockHistory)
      .catch((err) => console.warn(`Failed to fetch history for ${activeStock}:`, err))
      .finally(() => setHistoryLoading(false));
  }, [activeStock]);

  const sectors = ["ALL", ...Array.from(new Set(stocks.map((s) => s.sector)))];

  const filteredStocks = stocks.filter((s) => {
    const matchSearch =
      s.symbol.toLowerCase().includes(search.toLowerCase()) ||
      s.name.toLowerCase().includes(search.toLowerCase());
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
            INDIAN STOCK EXPLORER (NSE UNIVERSE)
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Canonical 10 Nifty 50 constituents with verified chronological market data feeds
          </p>
        </div>

        {/* Search & Sector Filters */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search symbol or company..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 pr-3 py-1.5 rounded-lg bg-[#11192d] border border-[#1f2d48] text-xs text-slate-100 placeholder:text-slate-400 focus:outline-none focus:border-teal-500/50 w-56"
            />
          </div>

          <select
            value={sectorFilter}
            onChange={(e) => setSectorFilter(e.target.value)}
            className="px-3 py-1.5 rounded-lg bg-[#11192d] border border-[#1f2d48] text-xs text-slate-300 font-mono focus:outline-none focus:border-teal-500/50"
          >
            {sectors.map((sec) => (
              <option key={sec} value={sec}>
                {sec}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 2-Column Layout: Stock Table + Detailed Chart Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Stock Table (7 cols) */}
        <div className="lg:col-span-7">
          <Card title="Constituent Securities" subtitle="Click any stock to inspect its full OHLCV series">
            {loading ? (
              <TableSkeleton rows={10} cols={4} />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#1b2844] text-slate-400">
                      <th className="py-2.5 px-3">SYMBOL</th>
                      <th className="py-2.5 px-3">COMPANY NAME</th>
                      <th className="py-2.5 px-3">SECTOR</th>
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
                          <td className="py-3 px-3 text-slate-200">{stock.name}</td>
                          <td className="py-3 px-3 text-slate-400 text-[11px]">{stock.sector}</td>
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
            title={`Detailed Analysis — ${activeStock || "Select Stock"}`}
            subtitle="Verified historical prices and exchange technicals"
            glow
          >
            {activeStock ? (
              <div className="space-y-4">
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

                {selectedStockHistory && (
                  <div className="pt-3 border-t border-[#1b2844] grid grid-cols-2 gap-3 text-xs font-mono">
                    <div className="p-2.5 rounded bg-[#0d1424] border border-[#1b2844]">
                      <span className="text-slate-400 text-[10px]">CURRENT PRICE</span>
                      <div className="text-base font-bold text-white mt-0.5">
                        ₹{selectedStockHistory.latest_price.toFixed(2)}
                      </div>
                    </div>
                    <div className="p-2.5 rounded bg-[#0d1424] border border-[#1b2844]">
                      <span className="text-slate-400 text-[10px]">FEED STATUS</span>
                      <div className="text-xs font-semibold text-teal-400 mt-1">
                        {selectedStockHistory.feed_status}
                      </div>
                    </div>
                  </div>
                )}

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
                    <span>View on NSE</span>
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
