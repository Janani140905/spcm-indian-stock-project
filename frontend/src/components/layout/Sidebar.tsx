import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  TrendingUp,
  BrainCircuit,
  Newspaper,
  Star,
  History,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";
import { useApp } from "../../context/AppContext.js";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/stocks", label: "Stock Explorer", icon: TrendingUp },
  { to: "/predictions", label: "ML Predictions", icon: BrainCircuit },
  { to: "/news", label: "News & Sentiment", icon: Newspaper },
  { to: "/watchlist", label: "Watchlist", icon: Star },
  { to: "/history", label: "Prediction History", icon: History },
  { to: "/methodology", label: "Methodology", icon: BookOpen },
];

export const Sidebar: React.FC = () => {
  const [collapsed, setCollapsed] = React.useState(false);
  const { demoMode } = useApp();

  return (
    <aside
      className={`h-[calc(100vh-4rem)] sticky top-16 bg-[#0e1526] border-r border-[#1b2844] flex flex-col justify-between transition-all duration-200 z-20 ${
        collapsed ? "w-16" : "w-64"
      }`}
    >
      {/* Top Nav links */}
      <div className="p-3 space-y-1.5">
        <div className="flex items-center justify-between px-3 py-2 text-[11px] font-mono uppercase tracking-wider text-slate-400">
          {!collapsed && <span>Terminal Navigation</span>}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors ml-auto"
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        <nav className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? "bg-gradient-to-r from-violet-950/80 to-teal-950/50 text-teal-300 border border-teal-500/30 shadow-sm"
                      : "text-slate-400 hover:text-slate-200 hover:bg-[#121c33]"
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0 text-slate-400 group-hover:text-teal-400" />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Bottom Footer Info */}
      <div className="p-3 border-t border-[#1b2844] bg-[#0b1020]/50">
        {demoMode && !collapsed && (
          <div className="mb-2 p-2 rounded bg-amber-950/80 border border-amber-600/50 text-[11px] text-amber-200 font-mono">
            ⚠ DEMO MODE ACTIVE
          </div>
        )}

        {!collapsed ? (
          <div className="space-y-1.5 text-[11px] font-mono text-slate-400">
            <div className="flex items-center justify-between text-slate-300 font-semibold">
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
                NSE 10-Stock Universe
              </span>
              <span className="text-teal-400">10 / 10</span>
            </div>
            <p className="text-[10px] text-slate-400 leading-relaxed">
              Reliance, TCS, HDFC Bank, Infosys, ICICI, HUL, ITC, SBI, Airtel, Kotak.
            </p>
          </div>
        ) : (
          <div className="flex justify-center text-teal-400 font-mono text-xs">10</div>
        )}
      </div>
    </aside>
  );
};
