import React from "react";
import {
  BookOpen,
  Layers,
  BrainCircuit,
  ShieldCheck,
  CheckCircle,
  Database,
  TrendingUp,
  Cpu,
} from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";

export const MethodologyPage: React.FC = () => {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="border-b border-[#1b2844] pb-4">
        <h1 className="text-xl font-bold font-mono text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-teal-400" />
          RESEARCH METHODOLOGY & SYSTEM ARCHITECTURE
        </h1>
        <p className="text-xs text-slate-400 mt-1 font-mono">
          Rigorous leak-audited multimodal Indian Stock Price Classification Model (SPCM)
        </p>
      </div>

      {/* Pipeline Diagram Strip */}
      <div className="p-4 rounded-xl bg-gradient-to-r from-violet-950/40 via-[#0e1628] to-teal-950/40 border border-[#2a3c5e] font-mono text-xs text-slate-300 space-y-3">
        <div className="text-teal-400 font-bold text-xs">PIPELINE EXECUTION FLOW</div>
        <div className="flex flex-wrap items-center gap-2 text-[11px]">
          <span className="px-2.5 py-1 rounded bg-[#11192d] border border-[#1f2d48] text-white">NSE Historical OHLCV</span>
          <span className="text-slate-400">→</span>
          <span className="px-2.5 py-1 rounded bg-[#11192d] border border-[#1f2d48] text-white">40 Price Lag Features</span>
          <span className="text-slate-400">→</span>
          <span className="px-2.5 py-1 rounded bg-[#11192d] border border-[#1f2d48] text-white">Indian News FinBERT</span>
          <span className="text-slate-400">→</span>
          <span className="px-2.5 py-1 rounded bg-[#11192d] border border-[#1f2d48] text-white">Sent / PV / AI Indices</span>
          <span className="text-slate-400">→</span>
          <span className="px-2.5 py-1 rounded bg-[#11192d] border border-[#1f2d48] text-teal-300 font-bold border-teal-500/40">45-D Fusion Matrix</span>
          <span className="text-slate-400">→</span>
          <span className="px-2.5 py-1 rounded bg-[#11192d] border border-[#1f2d48] text-violet-300 font-bold border-violet-500/40">MLP Classifier (64, 32)</span>
          <span className="text-slate-400">→</span>
          <span className="px-2.5 py-1 rounded bg-teal-950/80 border border-teal-500 text-teal-200 font-bold">Top-K Recommendations</span>
        </div>
      </div>

      {/* Grid of Core Methodological Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="1. Indian Market Universe" subtitle="Nifty 50 Large-Cap Equities">
          <p className="text-xs text-slate-300 leading-relaxed mb-3">
            The target market is strictly the National Stock Exchange of India (NSE). The canonical 10-stock universe comprises:
          </p>
          <div className="flex flex-wrap gap-1.5 font-mono text-xs">
            {["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK"].map((s) => (
              <span key={s} className="px-2 py-0.5 rounded bg-[#0b1020] border border-[#1f2d48] text-teal-300 font-semibold">
                {s}
              </span>
            ))}
          </div>
          <p className="text-[11px] text-slate-400 mt-3 font-mono">
            Yahoo tickers are normalized consistently (e.g. <code>RELIANCE.NS</code> ↔ <code>RELIANCE</code>).
          </p>
        </Card>

        <Card title="2. Technical Price Features" subtitle="Multi-lag feature generation without lookahead">
          <p className="text-xs text-slate-300 leading-relaxed">
            Constructs 10 base indicators over 4 chronological backward lags (1, 5, 10, 30 days):
          </p>
          <ul className="mt-2 space-y-1 text-xs text-slate-300 font-mono list-disc list-inside">
            <li>Return momentum: <code>ret_1d, ret_5d, ret_10d, ret_30d</code></li>
            <li>Moving average spreads: <code>close / sma_k - 1</code></li>
            <li>Realized volatility: <code>vol_k</code></li>
            <li>Volume ratios: <code>vol_ratio_k</code></li>
          </ul>
          <p className="text-[11px] text-slate-400 mt-2 font-mono">
            Requires minimum 60 trading days of historical warm-up data.
          </p>
        </Card>

        <Card title="3. Financial News Sentiment" subtitle="Point-in-time FinBERT & Lexicon fallback">
          <p className="text-xs text-slate-300 leading-relaxed">
            News is ingested through Google News RSS and Yahoo Finance. Articles after 15:30 IST market close roll into the following trading day to prevent lookahead contamination.
          </p>
          <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs font-mono">
            <div className="p-2 rounded bg-[#0b1020] border border-[#1f2d48]">
              <div className="text-teal-400 font-bold">Sent</div>
              <div className="text-[10px] text-slate-400">Net Sentiment [-1, 1]</div>
            </div>
            <div className="p-2 rounded bg-[#0b1020] border border-[#1f2d48]">
              <div className="text-teal-400 font-bold">PV</div>
              <div className="text-[10px] text-slate-400">Price-Volume interaction</div>
            </div>
            <div className="p-2 rounded bg-[#0b1020] border border-[#1f2d48]">
              <div className="text-teal-400 font-bold">AI</div>
              <div className="text-[10px] text-slate-400">Abnormal News Index</div>
            </div>
          </div>
        </Card>

        <Card title="4. Target Definition & Zero-Refit Guarantee" subtitle="Leakage prevention & model validation">
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong>Target:</strong> Binary label indicating membership in the top 30% of cross-sectional 10-day forward returns (<code className="text-teal-300">fwd_ret</code>).
          </p>
          <div className="mt-3 p-3 rounded bg-[#0b1020] border border-amber-600/40 text-[11px] text-amber-200 font-mono space-y-1">
            <div className="font-bold flex items-center gap-1.5 text-amber-400">
              <ShieldCheck className="w-4 h-4" />
              <span>ZERO-REFIT INFERENCE GUARANTEE</span>
            </div>
            <p>
              Inference uses <code>imputer.transform(X)</code> and <code>scaler.transform(X)</code> strictly.
              Transformers are never refit on live inference data, preserving frozen training parameters.
            </p>
          </div>
        </Card>
      </div>

      {/* PERN Architecture Specification */}
      <Card title="5. Full-Stack PERN Integration" subtitle="Enterprise architecture powering this platform">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-[#0b1020] border border-[#1f2d48]">
            <span className="text-teal-400 font-bold">FRONTEND</span>
            <div className="text-white font-semibold mt-1">React 18 + Vite</div>
            <p className="text-[10px] text-slate-400 mt-1">Tailwind CSS, Recharts, Lucide, TanStack Query, React Router</p>
          </div>

          <div className="p-3 rounded-lg bg-[#0b1020] border border-[#1f2d48]">
            <span className="text-violet-400 font-bold">BACKEND API</span>
            <div className="text-white font-semibold mt-1">Node.js + Express</div>
            <p className="text-[10px] text-slate-400 mt-1">TypeScript, Zod validation, Helmet, CORS, centralized errors</p>
          </div>

          <div className="p-3 rounded-lg bg-[#0b1020] border border-[#1f2d48]">
            <span className="text-indigo-400 font-bold">DATABASE</span>
            <div className="text-white font-semibold mt-1">PostgreSQL + Prisma</div>
            <p className="text-[10px] text-slate-400 mt-1">Typed ORM, Watchlists, PredictionHistory audit logs, migrations</p>
          </div>

          <div className="p-3 rounded-lg bg-[#0b1020] border border-[#1f2d48]">
            <span className="text-emerald-400 font-bold">ML ENGINE</span>
            <div className="text-white font-semibold mt-1">Python FastAPI (:8000)</div>
            <p className="text-[10px] text-slate-400 mt-1">SPCMInferencePipeline, joblib artifact bundle, yfinance, FinBERT</p>
          </div>
        </div>
      </Card>
    </div>
  );
};
