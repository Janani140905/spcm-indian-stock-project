import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Star, Trash2, TrendingUp, Plus, Edit2, ExternalLink } from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { TableSkeleton } from "../components/common/Skeleton.js";
import { useApp } from "../context/AppContext.js";
import {
  fetchWatchlist,
  removeFromWatchlist,
  addToWatchlist,
  fetchStockHistory,
} from "../services/api.js";
import { WatchlistItem, StockHistory } from "../types/index.js";

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

export const WatchlistPage: React.FC = () => {
  const { addToast, setActiveStock } = useApp();
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [stockPrices, setStockPrices] = useState<Record<string, { price: number; change: number }>>({});
  const [addSymbol, setAddSymbol] = useState("RELIANCE");
  const [targetPrice, setTargetPrice] = useState("");
  const [notes, setNotes] = useState("");

  const loadWatchlist = async () => {
    setLoading(true);
    try {
      const data = await fetchWatchlist();
      setItems(data);

      // Fetch prices for items
      const priceMap: Record<string, { price: number; change: number }> = {};
      await Promise.all(
        data.map(async (item) => {
          try {
            const h = await fetchStockHistory(item.symbol, "1mo");
            priceMap[item.symbol] = { price: h.latest_price, change: h.change_pct };
          } catch {
            // Price fetch fallback
          }
        })
      );
      setStockPrices(priceMap);
    } catch (err: any) {
      console.warn("Failed to load watchlist:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadWatchlist();
  }, []);

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const tPrice = targetPrice ? parseFloat(targetPrice) : undefined;
      const newItem = await addToWatchlist({
        symbol: addSymbol,
        targetPrice: tPrice,
        notes: notes || undefined,
      });
      setItems((prev) => [...prev.filter((i) => i.symbol !== newItem.symbol), newItem]);
      setNotes("");
      setTargetPrice("");
      addToast("Stock Added", `Added ${newItem.symbol} to your watchlist`, "success");
    } catch (err: any) {
      addToast("Add Failed", err.message || "Failed to add to watchlist", "error");
    }
  };

  const handleRemove = async (symbol: string) => {
    try {
      await removeFromWatchlist(symbol);
      setItems((prev) => prev.filter((i) => i.symbol !== symbol));
      addToast("Stock Removed", `Removed ${symbol} from your watchlist`, "info");
    } catch (err: any) {
      addToast("Remove Failed", err.message || "Failed to remove from watchlist", "error");
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1b2844] pb-4">
        <div>
          <h1 className="text-xl font-bold font-mono text-white flex items-center gap-2">
            <Star className="w-5 h-5 text-amber-400 fill-amber-400" />
            PERSISTENT STOCK WATCHLIST
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Track key Indian equities, target buy/sell price benchmarks, and strategy notes
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Watchlist Table (2 cols) */}
        <div className="lg:col-span-2 space-y-4">
          <Card
            title={`Watched Equities (${items.length} stocks)`}
            subtitle="Real-time / delayed price updates and target alert tracking"
          >
            {loading ? (
              <TableSkeleton rows={4} cols={5} />
            ) : items.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs font-mono">
                  <thead>
                    <tr className="border-b border-[#1b2844] text-slate-400">
                      <th className="py-2.5 px-3">STOCK</th>
                      <th className="py-2.5 px-3">LATEST PRICE</th>
                      <th className="py-2.5 px-3">TARGET BENCHMARK</th>
                      <th className="py-2.5 px-3">NOTES</th>
                      <th className="py-2.5 px-3 text-right">ACTION</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#16223b]">
                    {items.map((item) => {
                      const quote = stockPrices[item.symbol];
                      const isUp = quote ? quote.change >= 0 : true;
                      return (
                        <tr key={item.symbol} className="hover:bg-[#141e35] transition-colors">
                          <td className="py-3 px-3 font-bold text-white">
                            <Link
                              to="/stocks"
                              onClick={() => setActiveStock(item.symbol)}
                              className="hover:text-teal-300 flex items-center gap-1.5"
                            >
                              <span>{item.symbol}</span>
                              <span className="text-[10px] text-slate-400 font-normal">.NS</span>
                            </Link>
                          </td>
                          <td className="py-3 px-3">
                            {quote ? (
                              <div className="flex items-center gap-2">
                                <span className="font-semibold text-white">
                                  ₹{quote.price.toFixed(2)}
                                </span>
                                <span
                                  className={`text-[10px] ${
                                    isUp ? "text-emerald-400" : "text-rose-400"
                                  }`}
                                >
                                  {isUp ? "+" : ""}
                                  {quote.change.toFixed(2)}%
                                </span>
                              </div>
                            ) : (
                              <span className="text-slate-400">—</span>
                            )}
                          </td>
                          <td className="py-3 px-3 text-slate-300">
                            {item.targetPrice ? `₹${item.targetPrice.toFixed(2)}` : "None set"}
                          </td>
                          <td className="py-3 px-3 text-slate-400 max-w-[150px] truncate">
                            {item.notes || "—"}
                          </td>
                          <td className="py-3 px-3 text-right">
                            <button
                              onClick={() => handleRemove(item.symbol)}
                              className="p-1.5 rounded text-rose-400 hover:text-rose-300 hover:bg-rose-950/40 transition-colors"
                              title={`Remove ${item.symbol}`}
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-12 text-center text-xs font-mono text-slate-400 space-y-2">
                <Star className="w-8 h-8 text-slate-400 mx-auto" />
                <p>Your watchlist is currently empty.</p>
                <p className="text-[11px] text-slate-400">
                  Use the quick form on the right or the Stock Explorer to add stocks.
                </p>
              </div>
            )}
          </Card>
        </div>

        {/* Right Column: Quick Add Stock Form */}
        <div>
          <Card
            title="Add Security to Watchlist"
            subtitle="Configure target price and custom investment thesis"
            glow
          >
            <form onSubmit={handleAdd} className="space-y-4 text-xs font-mono">
              <div>
                <label className="text-slate-400 block mb-1">SELECT STOCK:</label>
                <select
                  value={addSymbol}
                  onChange={(e) => setAddSymbol(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[#0b1020] border border-[#1f2d48] text-slate-200 focus:outline-none focus:border-teal-500/50"
                >
                  {ALL_SYMBOLS.map((sym) => (
                    <option key={sym} value={sym}>
                      {sym}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-slate-400 block mb-1">TARGET PRICE BENCHMARK (₹, OPTIONAL):</label>
                <input
                  type="number"
                  placeholder="e.g. 1350.00"
                  value={targetPrice}
                  onChange={(e) => setTargetPrice(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[#0b1020] border border-[#1f2d48] text-slate-200 focus:outline-none focus:border-teal-500/50"
                />
              </div>

              <div>
                <label className="text-slate-400 block mb-1">RESEARCH NOTES (OPTIONAL):</label>
                <textarea
                  rows={3}
                  placeholder="e.g. SPCM top-3 recommendation; watch 10-day forward return."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-[#0b1020] border border-[#1f2d48] text-slate-200 focus:outline-none focus:border-teal-500/50"
                />
              </div>

              <button
                type="submit"
                className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white font-bold transition-colors shadow-lg shadow-teal-500/20"
              >
                <Plus className="w-4 h-4" />
                <span>SAVE TO WATCHLIST</span>
              </button>
            </form>
          </Card>
        </div>
      </div>
    </div>
  );
};
