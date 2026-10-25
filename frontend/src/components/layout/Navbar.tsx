import React from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Search,
  RefreshCw,
  Sliders,
  Database,
  Cpu,
  Server,
  AlertTriangle,
} from "lucide-react";
import { useApp } from "../../context/AppContext.js";
import { Badge } from "../common/Badge.js";

export const Navbar: React.FC = () => {
  const { demoMode, setDemoMode, setSearchOpen, health, refreshHealth, addToast } = useApp();
  const [refreshing, setRefreshing] = React.useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    await refreshHealth();
    setTimeout(() => {
      setRefreshing(false);
      addToast("System Refreshed", "Latest backend and feed health status updated", "info");
    }, 600);
  };

  const isPyOnline = health?.services?.python_ml_engine?.status === "up";
  const isDbLive = health?.services?.database?.mode === "live";

  return (
    <header className="sticky top-0 z-30 h-16 bg-[#0b1020]/90 backdrop-blur-md border-b border-[#1b2844] px-6 flex items-center justify-between">
      {/* Brand & Market Status */}
      <div className="flex items-center gap-4">
        <Link to="/" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 via-indigo-600 to-teal-400 p-[1px] flex items-center justify-center shadow-lg shadow-violet-500/20">
            <div className="w-full h-full bg-[#0b1020] rounded-[7px] flex items-center justify-center">
              <Activity className="w-4 h-4 text-teal-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white font-mono">
                SPCM <span className="text-teal-400">INTELLIGENCE</span>
              </span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-violet-950/90 text-violet-300 border border-violet-500/30 font-mono">
                v1.0
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono tracking-wider">
              INDIAN STOCK PREDICTION PLATFORM (NSE)
            </p>
          </div>
        </Link>

        {/* Live feed badge */}
        <div className="hidden md:flex items-center gap-2 pl-4 border-l border-[#1b2844]">
          <span className="pulse-live" />
          <span className="text-xs text-slate-300 font-mono">
            NSE DATA: <span className="text-teal-300 font-semibold">15m DELAYED / EOD</span>
          </span>
        </div>
      </div>

      {/* Center Quick Search Trigger */}
      <div className="hidden lg:flex items-center">
        <button
          onClick={() => setSearchOpen(true)}
          className="flex items-center gap-3 px-3.5 py-1.5 rounded-lg bg-[#11192d] border border-[#1f2d48] text-xs text-slate-400 hover:border-teal-500/50 hover:text-slate-200 transition-all shadow-inner"
        >
          <Search className="w-3.5 h-3.5 text-teal-400" />
          <span>Search 10 NSE Stocks (e.g. RELIANCE, TCS)...</span>
          <kbd className="px-1.5 py-0.5 text-[10px] bg-slate-800 text-slate-400 rounded font-mono border border-slate-700">
            Ctrl+K
          </kbd>
        </button>
      </div>

      {/* Right controls & Service status */}
      <div className="flex items-center gap-3">
        {/* Service health indicators */}
        <div className="hidden xl:flex items-center gap-2 text-xs font-mono">
          <div
            className={`flex items-center gap-1 px-2 py-1 rounded border ${
              isPyOnline
                ? "bg-emerald-950/50 border-emerald-500/30 text-emerald-300"
                : "bg-amber-950/50 border-amber-500/30 text-amber-300"
            }`}
            title={
              isPyOnline
                ? "Python ML Service online (FastAPI :8000)"
                : "Python ML Service offline"
            }
          >
            <Cpu className="w-3 h-3" />
            <span>ML: {isPyOnline ? "LIVE" : "OFFLINE"}</span>
          </div>

          <div
            className={`flex items-center gap-1 px-2 py-1 rounded border ${
              isDbLive
                ? "bg-emerald-950/50 border-emerald-500/30 text-emerald-300"
                : "bg-slate-800/60 border-slate-700 text-slate-400"
            }`}
            title={
              isDbLive
                ? "PostgreSQL live database connected"
                : "PostgreSQL offline; using local in-memory persistence"
            }
          >
            <Database className="w-3 h-3" />
            <span>DB: {isDbLive ? "POSTGRES" : "MEMORY"}</span>
          </div>
        </div>

        {/* Demo Mode Toggle */}
        <button
          onClick={() => {
            const nextMode = !demoMode;
            setDemoMode(nextMode);
            addToast(
              nextMode ? "Demo Mode Active" : "Real-Data Mode Active",
              nextMode
                ? "Displaying isolated illustrative mock data with persistent DEMO badges."
                : "Fetching genuine quotes, live news sentiment, and frozen model predictions.",
              nextMode ? "warning" : "success"
            );
          }}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-mono font-medium border transition-all ${
            demoMode
              ? "bg-amber-900/60 text-amber-200 border-amber-600 shadow-sm shadow-amber-500/20"
              : "bg-[#11192d] text-slate-300 border-[#1f2d48] hover:border-slate-600"
          }`}
          title="Toggle between Real Pipeline Data and Isolated Demo Mode"
        >
          <Sliders className="w-3.5 h-3.5 text-amber-400" />
          <span>{demoMode ? "DEMO MODE ON" : "REAL DATA"}</span>
        </button>

        {/* Refresh button */}
        <button
          onClick={handleRefresh}
          disabled={refreshing}
          className="p-2 rounded-lg bg-[#11192d] border border-[#1f2d48] text-slate-400 hover:text-teal-300 hover:border-teal-500/40 transition-colors"
          title="Refresh connection and data feeds"
        >
          <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin text-teal-400" : ""}`} />
        </button>
      </div>
    </header>
  );
};
