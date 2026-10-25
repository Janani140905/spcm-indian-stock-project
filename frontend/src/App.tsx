import React from "react";
import { Routes, Route } from "react-router-dom";
import { Navbar } from "./components/layout/Navbar.js";
import { Sidebar } from "./components/layout/Sidebar.js";
import { SymbolSearchModal } from "./components/common/SymbolSearchModal.js";
import { useApp } from "./context/AppContext.js";
import { OverviewPage } from "./pages/OverviewPage.js";
import { StockExplorerPage } from "./pages/StockExplorerPage.js";
import { PredictionsPage } from "./pages/PredictionsPage.js";
import { NewsSentimentPage } from "./pages/NewsSentimentPage.js";
import { WatchlistPage } from "./pages/WatchlistPage.js";
import { PredictionHistoryPage } from "./pages/PredictionHistoryPage.js";
import { MethodologyPage } from "./pages/MethodologyPage.js";
import { X, CheckCircle, AlertCircle, Info } from "lucide-react";

export const App: React.FC = () => {
  const { toasts, removeToast } = useApp();

  return (
    <div className="min-h-screen bg-[#0b1020] text-slate-100 flex flex-col font-sans">
      <Navbar />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar />

        <main className="flex-1 overflow-y-auto p-6 md:p-8">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/stocks" element={<StockExplorerPage />} />
            <Route path="/predictions" element={<PredictionsPage />} />
            <Route path="/news" element={<NewsSentimentPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/history" element={<PredictionHistoryPage />} />
            <Route path="/methodology" element={<MethodologyPage />} />
            {/* Fallback route */}
            <Route path="*" element={<OverviewPage />} />
          </Routes>
        </main>
      </div>

      {/* Global Quick Search Modal */}
      <SymbolSearchModal />

      {/* Toast Notification Container */}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
        {toasts.map((toast) => {
          const icon =
            toast.type === "success" ? (
              <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
            ) : toast.type === "error" ? (
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            ) : toast.type === "warning" ? (
              <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
            ) : (
              <Info className="w-4 h-4 text-teal-400 shrink-0" />
            );

          const borderClass =
            toast.type === "success"
              ? "border-emerald-500/40 bg-[#0d1c24]"
              : toast.type === "error"
              ? "border-rose-500/40 bg-[#1c0d15]"
              : toast.type === "warning"
              ? "border-amber-500/40 bg-[#1c180d]"
              : "border-teal-500/40 bg-[#0d1726]";

          return (
            <div
              key={toast.id}
              className={`p-3.5 rounded-xl border shadow-2xl backdrop-blur-md pointer-events-auto flex items-start gap-3 text-xs animate-in slide-in-from-bottom-2 ${borderClass}`}
            >
              <div className="mt-0.5">{icon}</div>
              <div className="flex-1">
                <div className="font-bold text-white font-mono">{toast.title}</div>
                <div className="text-slate-300 mt-0.5 leading-relaxed">{toast.message}</div>
              </div>
              <button
                onClick={() => removeToast(toast.id)}
                className="text-slate-400 hover:text-white p-0.5"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default App;
