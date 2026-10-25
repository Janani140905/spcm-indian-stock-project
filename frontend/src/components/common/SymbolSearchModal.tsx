import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Search, X, TrendingUp } from "lucide-react";
import { useApp } from "../../context/AppContext.js";

const CANONICAL_STOCKS = [
  { symbol: "RELIANCE", name: "Reliance Industries Ltd", sector: "Energy & Conglomerate" },
  { symbol: "TCS", name: "Tata Consultancy Services Ltd", sector: "Information Technology" },
  { symbol: "HDFCBANK", name: "HDFC Bank Ltd", sector: "Banking & Financial Services" },
  { symbol: "INFY", name: "Infosys Ltd", sector: "Information Technology" },
  { symbol: "ICICIBANK", name: "ICICI Bank Ltd", sector: "Banking & Financial Services" },
  { symbol: "HINDUNILVR", name: "Hindustan Unilever Ltd", sector: "Consumer Goods (FMCG)" },
  { symbol: "ITC", name: "ITC Ltd", sector: "Consumer Goods & Conglomerate" },
  { symbol: "SBIN", name: "State Bank of India", sector: "Public Sector Banking" },
  { symbol: "BHARTIARTL", name: "Bharti Airtel Ltd", sector: "Telecommunications" },
  { symbol: "KOTAKBANK", name: "Kotak Mahindra Bank Ltd", sector: "Banking & Financial Services" },
];

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

  const filtered = CANONICAL_STOCKS.filter(
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
            placeholder="Search stock symbol or company name..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoFocus
            className="w-full bg-transparent text-sm text-slate-100 placeholder:text-slate-400 focus:outline-none"
          />
          <button
            onClick={() => setSearchOpen(false)}
            className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto p-2 divide-y divide-[#18233a]">
          {filtered.length > 0 ? (
            filtered.map((stock) => (
              <button
                key={stock.symbol}
                onClick={() => handleSelect(stock.symbol)}
                className="w-full text-left px-3 py-2.5 rounded-lg hover:bg-[#16223b] flex items-center justify-between transition-colors group"
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold text-sm text-white group-hover:text-teal-300">
                      {stock.symbol}
                    </span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                      NSE
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 mt-0.5">{stock.name}</div>
                </div>
                <div className="text-right">
                  <span className="text-[11px] text-slate-400">{stock.sector}</span>
                  <div className="flex items-center justify-end gap-1 text-teal-400 text-xs mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <span>Analyze</span>
                    <TrendingUp className="w-3 h-3" />
                  </div>
                </div>
              </button>
            ))
          ) : (
            <div className="p-6 text-center text-xs text-slate-400">
              No matching Indian stocks found for "{query}".
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-[#1b2844] bg-[#0c1322] flex items-center justify-between text-[11px] font-mono text-slate-400">
          <span>Supported: 10 Indian Large-Cap NSE Stocks</span>
          <span>Press ESC to close</span>
        </div>
      </div>
    </div>
  );
};
