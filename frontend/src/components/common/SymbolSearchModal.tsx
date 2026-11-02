import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Search, X, TrendingUp, Layers } from "lucide-react";
import { useApp } from "../../context/AppContext.js";
import { STOCK_UNIVERSE } from "../../config/stockUniverse.js";

export const SymbolSearchModal: React.FC = () => {
  const { searchOpen, setSearchOpen, setActiveStock } = useApp();
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen(true);
      } else if (e.key === "Escape") {
        setSearchOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [setSearchOpen]);

  if (!searchOpen) return null;

  const filtered = STOCK_UNIVERSE.filter(
    (s) =>
      s.symbol.toLowerCase().includes(query.toLowerCase()) ||
      s.name.toLowerCase().includes(query.toLowerCase()) ||
      s.sector.toLowerCase().includes(query.toLowerCase())
  );

  const handleSelect = (symbol: string) => {
    setActiveStock(symbol);
    setSearchOpen(false);
    navigate("/stocks");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 bg-black/70 backdrop-blur-sm p-4 animate-in fade-in duration-150">
      <div className="w-full max-w-lg bg-[#11192d] border border-[#2a3c5e] rounded-xl shadow-2xl overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-[#1b2844] bg-[#0e1628]">
          <Search className="w-5 h-5 text-teal-400 shrink-0" />
          <input
            type="text"
            placeholder="Search across 50 Indian stocks or 12 sectors (e.g. SUNPHARMA, MARUTI, Pharma)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
            className="w-full bg-transparent text-sm text-slate-100 placeholder:text-slate-400 focus:outline-none font-sans"
          />
          <button
            onClick={() => setSearchOpen(false)}
            className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="max-h-80 overflow-y-auto divide-y divide-[#1b2844]">
          {filtered.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-400 font-mono">
              No matching Indian stock symbols found for "{query}".
            </div>
          ) : (
            filtered.map((stock) => (
              <button
                key={stock.symbol}
                onClick={() => handleSelect(stock.symbol)}
                className="w-full px-4 py-3 text-left hover:bg-[#16223b] flex items-center justify-between transition-colors group"
              >
                <div className="flex items-center gap-3">
                  <div className="w-7 h-7 rounded-lg bg-teal-500/10 border border-teal-500/30 flex items-center justify-center text-teal-400 text-xs font-mono font-bold group-hover:bg-teal-500 group-hover:text-black transition-colors">
                    {stock.symbol.slice(0, 2)}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-white text-xs">
                        {stock.symbol}
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">.NS</span>
                      {stock.inTrainingUniverse && (
                        <span className="px-1.5 py-0.2 rounded bg-violet-950 text-violet-300 text-[9px] border border-violet-500/40">
                          SPCM Benchmark
                        </span>
                      )}
                    </div>
                    <div className="text-[11px] text-slate-400 truncate max-w-xs">
                      {stock.name}
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-[#0b1020] text-slate-300 border border-[#1b2844]">
                    {stock.sector}
                  </span>
                </div>
              </button>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-[#1b2844] bg-[#0b1020] flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>{filtered.length} of 50 stocks available across 12 sectors</span>
          <span>ESC to close</span>
        </div>
      </div>
    </div>
  );
};
