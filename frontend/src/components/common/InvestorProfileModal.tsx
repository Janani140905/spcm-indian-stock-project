import React, { useState, useEffect } from "react";
import {
  X,
  Shield,
  Zap,
  TrendingUp,
  Target,
  Sparkles,
  Check,
  RotateCcw,
  Clock,
  Newspaper,
  Compass,
  Wallet,
  DollarSign,
  Briefcase,
} from "lucide-react";
import { useApp } from "../../context/AppContext.js";
import {
  AVAILABLE_SECTORS,
  DEFAULT_INVESTOR_PROFILE,
  saveStoredProfile,
} from "../../services/personalization.js";
import { saveProfile as saveBackendProfile } from "../../services/api.js";
import {
  InvestorProfile,
  RiskTolerance,
  InvestmentHorizon,
  InvestmentObjective,
  BudgetRange,
  SentimentPreference,
} from "../../types/index.js";

export const InvestorProfileModal: React.FC = () => {
  const { profileModalOpen, setProfileModalOpen, investorProfile, setInvestorProfile, addToast } = useApp();

  const [riskTolerance, setRiskTolerance] = useState<RiskTolerance>("moderate");
  const [preferredSectors, setPreferredSectors] = useState<string[]>(DEFAULT_INVESTOR_PROFILE.preferredSectors);
  const [investmentHorizon, setInvestmentHorizon] = useState<InvestmentHorizon>("medium_term");
  const [investmentObjective, setInvestmentObjective] = useState<InvestmentObjective>("balanced");
  const [budgetRange, setBudgetRange] = useState<BudgetRange>("25k_to_100k");
  const [sentimentPreference, setSentimentPreference] = useState<SentimentPreference>("balanced");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (investorProfile) {
      setRiskTolerance(investorProfile.riskTolerance);
      setPreferredSectors(investorProfile.preferredSectors);
      setInvestmentHorizon(investorProfile.investmentHorizon);
      setInvestmentObjective(investorProfile.investmentObjective || "balanced");
      setBudgetRange(investorProfile.budgetRange || "25k_to_100k");
      setSentimentPreference(investorProfile.sentimentPreference);
    }
  }, [investorProfile, profileModalOpen]);

  if (!profileModalOpen) return null;

  const toggleSector = (sector: string) => {
    if (preferredSectors.includes(sector)) {
      if (preferredSectors.length <= 1) {
        addToast("Sector Notice", "Please keep at least 1 sector selected.", "warning");
        return;
      }
      setPreferredSectors((prev) => prev.filter((s) => s !== sector));
    } else {
      setPreferredSectors((prev) => [...prev, sector]);
    }
  };

  const selectAllSectors = () => {
    setPreferredSectors(AVAILABLE_SECTORS.map((s) => s.id));
  };

  const handleSave = async () => {
    setSaving(true);
    const updated: InvestorProfile = {
      riskTolerance,
      preferredSectors,
      investmentHorizon,
      investmentObjective,
      budgetRange,
      sentimentPreference,
      updatedAt: new Date().toISOString(),
    };

    saveStoredProfile(updated);
    setInvestorProfile(updated);

    try {
      await saveBackendProfile(updated);
    } catch (e) {
      // Local storage saved successfully even if network hiccup occurs
    } finally {
      setSaving(false);
      setProfileModalOpen(false);
      addToast(
        "Profile Updated",
        "Stock recommendations across all 12 Indian sectors updated to match your tastes!",
        "success"
      );
    }
  };

  const handleReset = () => {
    setRiskTolerance(DEFAULT_INVESTOR_PROFILE.riskTolerance);
    setPreferredSectors(DEFAULT_INVESTOR_PROFILE.preferredSectors);
    setInvestmentHorizon(DEFAULT_INVESTOR_PROFILE.investmentHorizon);
    setInvestmentObjective(DEFAULT_INVESTOR_PROFILE.investmentObjective);
    setBudgetRange(DEFAULT_INVESTOR_PROFILE.budgetRange);
    setSentimentPreference(DEFAULT_INVESTOR_PROFILE.sentimentPreference);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150 overflow-y-auto">
      <div className="w-full max-w-3xl bg-[#0f172a] border border-[#233554] rounded-2xl shadow-2xl overflow-hidden my-6">
        {/* Header */}
        <div className="px-6 py-4 bg-[#0b1120] border-b border-[#1e293b] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-violet-600 via-indigo-600 to-teal-400 text-white shadow-lg shadow-violet-500/20">
              <Compass className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold font-mono text-white flex items-center gap-2">
                PERSONALIZED INVESTOR PROFILE QUESTIONNAIRE
                <span className="text-[10px] px-2 py-0.5 rounded bg-teal-950 text-teal-300 border border-teal-500/40">
                  12 Sectors · 50 Stocks
                </span>
              </h2>
              <p className="text-xs text-slate-400 font-mono">
                Configure your risk, sectors of interest, and objectives for personalized stock recommendations
              </p>
            </div>
          </div>
          <button
            onClick={() => setProfileModalOpen(false)}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6 max-h-[75vh] overflow-y-auto font-sans">
          {/* Question 1: Investment Objective */}
          <div className="space-y-3">
            <label className="text-xs font-mono font-bold text-slate-300 flex items-center gap-2">
              <Target className="w-4 h-4 text-teal-400" />
              1. WHAT IS YOUR PRIMARY INVESTMENT OBJECTIVE?
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5">
              {[
                { id: "capital_growth" as InvestmentObjective, title: "Capital Growth", desc: "Maximize capital appreciation via high-momentum leaders" },
                { id: "wealth_preservation" as InvestmentObjective, title: "Wealth Preservation", desc: "Protect capital with low-volatility bluechip defensives" },
                { id: "dividend_income" as InvestmentObjective, title: "Dividend Income", desc: "Target high cash-flow sectors with steady yield" },
                { id: "balanced" as InvestmentObjective, title: "Balanced Growth", desc: "Balanced blend of steady capital gains and stability" },
              ].map((opt) => {
                const isSelected = investmentObjective === opt.id;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setInvestmentObjective(opt.id)}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      isSelected
                        ? "bg-teal-950/40 border-teal-400 text-white shadow-sm"
                        : "bg-[#0b1020] border-[#1e293b] text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-xs font-mono">{opt.title}</span>
                      {isSelected && <Check className="w-3.5 h-3.5 text-teal-400 stroke-[3]" />}
                    </div>
                    <p className="text-[11px] text-slate-400">{opt.desc}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Question 2: Risk Tolerance */}
          <div className="space-y-3">
            <label className="text-xs font-mono font-bold text-slate-300 flex items-center gap-2">
              <Shield className="w-4 h-4 text-violet-400" />
              2. WHAT IS YOUR RISK TOLERANCE?
            </label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {[
                {
                  id: "conservative" as RiskTolerance,
                  title: "Conservative",
                  desc: "Prioritize low volatility (<22%) and defensive cashflows (HUL, ITC, Sun Pharma, TCS)",
                  icon: Shield,
                },
                {
                  id: "moderate" as RiskTolerance,
                  title: "Moderate",
                  desc: "Controlled market beta across diversified large-cap leaders (HDFC, Reliance, L&T)",
                  icon: Target,
                },
                {
                  id: "aggressive" as RiskTolerance,
                  title: "Aggressive",
                  desc: "Seek high beta, momentum breakouts and catalyst upside (Auto/EV, Tech, Trent)",
                  icon: Zap,
                },
              ].map((opt) => {
                const isSelected = riskTolerance === opt.id;
                const Icon = opt.icon;
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => setRiskTolerance(opt.id)}
                    className={`p-3.5 rounded-xl border text-left transition-all relative ${
                      isSelected
                        ? "bg-violet-950/60 border-teal-400 text-white shadow-lg shadow-teal-500/10"
                        : "bg-[#0b1020] border-[#1e293b] text-slate-300 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        <Icon className={`w-4 h-4 ${isSelected ? "text-teal-400" : "text-slate-400"}`} />
                        <span className="font-bold text-xs font-mono">{opt.title}</span>
                      </div>
                      {isSelected && (
                        <span className="w-4 h-4 rounded-full bg-teal-400 text-black flex items-center justify-center text-[10px]">
                          <Check className="w-3 h-3 stroke-[3]" />
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed">{opt.desc}</p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Question 3: Preferred Sectors (12 Indian Sectors) */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-mono font-bold text-slate-300 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                3. WHICH SECTORS ARE YOU INTERESTED IN? (12 SECTORS)
              </label>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={selectAllSectors}
                  className="text-[11px] font-mono text-teal-400 hover:underline"
                >
                  Select All
                </button>
                <span className="text-[11px] font-mono text-slate-400">
                  {preferredSectors.length} of {AVAILABLE_SECTORS.length} selected
                </span>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
              {AVAILABLE_SECTORS.map((sec) => {
                const isSelected = preferredSectors.includes(sec.id);
                return (
                  <button
                    key={sec.id}
                    type="button"
                    onClick={() => toggleSector(sec.id)}
                    className={`p-2.5 rounded-xl border text-left flex items-start justify-between transition-all ${
                      isSelected
                        ? "bg-teal-950/30 border-teal-500/60 text-white shadow-sm"
                        : "bg-[#0b1020] border-[#1e293b] text-slate-400 hover:border-slate-700 hover:text-slate-200"
                    }`}
                  >
                    <div className="pr-2">
                      <div className="text-xs font-semibold font-mono text-slate-200">{sec.label}</div>
                      <div className="text-[10px] text-slate-400 mt-0.5 line-clamp-1 font-mono">
                        {sec.stocks.slice(0, 3).join(", ")}
                        {sec.stocks.length > 3 ? ` +${sec.stocks.length - 3}` : ""}
                      </div>
                    </div>
                    <div
                      className={`w-4 h-4 mt-0.5 rounded border flex items-center justify-center shrink-0 transition-colors ${
                        isSelected
                          ? "bg-teal-500 border-teal-400 text-black"
                          : "border-slate-700 bg-slate-900"
                      }`}
                    >
                      {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Question 4: Investment Horizon & Budget Range */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Horizon */}
            <div className="space-y-3">
              <label className="text-xs font-mono font-bold text-slate-300 flex items-center gap-2">
                <Clock className="w-4 h-4 text-amber-400" />
                4. WHAT IS YOUR INVESTMENT HORIZON?
              </label>
              <div className="space-y-2">
                {[
                  { id: "short_term" as InvestmentHorizon, title: "Short-Term / Swing (Days to 2 Wks)", desc: "Heavy emphasis on news catalysts & 10d momentum" },
                  { id: "medium_term" as InvestmentHorizon, title: "Medium-Term (10-Day to 1 Mo)", desc: "Aligns with SPCM's native 10-day forward return window" },
                  { id: "long_term" as InvestmentHorizon, title: "Long-Term (Months+)", desc: "Focus on fundamental durability & large-cap resilience" },
                ].map((opt) => {
                  const isSelected = investmentHorizon === opt.id;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setInvestmentHorizon(opt.id)}
                      className={`w-full p-2.5 rounded-xl border text-left transition-all ${
                        isSelected
                          ? "bg-amber-950/40 border-amber-400 text-white"
                          : "bg-[#0b1020] border-[#1e293b] text-slate-300 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs font-mono">{opt.title}</span>
                        {isSelected && <Check className="w-3.5 h-3.5 text-amber-400 stroke-[3]" />}
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">{opt.desc}</p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Budget Range */}
            <div className="space-y-3">
              <label className="text-xs font-mono font-bold text-slate-300 flex items-center gap-2">
                <Wallet className="w-4 h-4 text-indigo-400" />
                5. ESTIMATED INVESTMENT BUDGET RANGE
              </label>
              <div className="space-y-2">
                {[
                  { id: "under_25k" as BudgetRange, title: "Starter (Under ₹25,000)", desc: "Flags concentration risk on high per-share stocks" },
                  { id: "25k_to_100k" as BudgetRange, title: "Core Portfolio (₹25,000 - ₹1,00,000)", desc: "Suitable for multi-stock diversification across sectors" },
                  { id: "100k_to_500k" as BudgetRange, title: "Substantial (₹1,00,000 - ₹5,00,000)", desc: "Balanced portfolio across 8-12 equities" },
                  { id: "above_500k" as BudgetRange, title: "High Net Worth (Above ₹5,00,000)", desc: "Full coverage across all 12 Indian industry sectors" },
                ].map((opt) => {
                  const isSelected = budgetRange === opt.id;
                  return (
                    <button
                      key={opt.id}
                      type="button"
                      onClick={() => setBudgetRange(opt.id)}
                      className={`w-full p-2.5 rounded-xl border text-left transition-all ${
                        isSelected
                          ? "bg-indigo-950/40 border-indigo-400 text-white"
                          : "bg-[#0b1020] border-[#1e293b] text-slate-300 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs font-mono">{opt.title}</span>
                        {isSelected && <Check className="w-3.5 h-3.5 text-indigo-400 stroke-[3]" />}
                      </div>
                      <p className="text-[10px] text-slate-400 mt-0.5">{opt.desc}</p>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-[#0b1120] border-t border-[#1e293b] flex items-center justify-between">
          <button
            type="button"
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-mono text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defaults</span>
          </button>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setProfileModalOpen(false)}
              className="px-4 py-2 rounded-lg text-xs font-mono text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={saving}
              className="flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-violet-600 via-indigo-600 to-teal-500 text-white font-mono text-xs font-bold hover:opacity-90 shadow-lg shadow-violet-500/25 transition-all disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{saving ? "SAVING..." : "SAVE & RE-RANK RECOMMENDATIONS"}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
