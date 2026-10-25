import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { History, Filter, Database, ArrowRight, ShieldCheck } from "lucide-react";
import { Card } from "../components/common/Card.js";
import { Badge } from "../components/common/Badge.js";
import { TableSkeleton } from "../components/common/Skeleton.js";
import { fetchPredictionHistory } from "../services/api.js";
import { PredictionHistoryRecord } from "../types/index.js";

const SYMBOL_OPTIONS = [
  "ALL",
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

export const PredictionHistoryPage: React.FC = () => {
  const [records, setRecords] = useState<PredictionHistoryRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [source, setSource] = useState("postgresql");
  const [loading, setLoading] = useState(true);
  const [stockFilter, setStockFilter] = useState("ALL");
  const [limit] = useState(25);
  const [page, setPage] = useState(0);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const stockParam = stockFilter !== "ALL" ? stockFilter : undefined;
      const res = await fetchPredictionHistory({
        stock: stockParam,
        limit,
        offset: page * limit,
      });
      setRecords(res.records);
      setTotal(res.total);
      setSource(res.source);
    } catch (err: any) {
      console.warn("Failed to load prediction history:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [stockFilter, page]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#1b2844] pb-4">
        <div>
          <h1 className="text-xl font-bold font-mono text-white flex items-center gap-2">
            <History className="w-5 h-5 text-teal-400" />
            HISTORICAL PREDICTION AUDIT LOG
          </h1>
          <p className="text-xs text-slate-400 mt-1 font-mono">
            Immutable log of genuine inference predictions persisted via PostgreSQL & Prisma
          </p>
        </div>

        {/* Database mode badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#11192d] border border-[#1f2d48] text-xs font-mono">
          <Database className="w-3.5 h-3.5 text-teal-400" />
          <span className="text-slate-400">STORE:</span>
          <span className="text-teal-300 font-bold uppercase">{source}</span>
        </div>
      </div>

      {/* Filter toolbar */}
      <div className="flex items-center justify-between gap-4 p-3 rounded-xl bg-[#11192d] border border-[#1b2844]">
        <div className="flex items-center gap-3 text-xs font-mono">
          <Filter className="w-3.5 h-3.5 text-teal-400" />
          <span className="text-slate-400">FILTER BY STOCK:</span>
          <select
            value={stockFilter}
            onChange={(e) => {
              setStockFilter(e.target.value);
              setPage(0);
            }}
            className="px-2.5 py-1 rounded bg-[#0b1020] border border-[#1f2d48] text-slate-200 focus:outline-none"
          >
            {SYMBOL_OPTIONS.map((sym) => (
              <option key={sym} value={sym}>
                {sym}
              </option>
            ))}
          </select>
        </div>

        <div className="text-xs font-mono text-slate-400">
          Showing {records.length} of {total} genuine records
        </div>
      </div>

      {/* History Table */}
      <Card title="Inference Audit History" subtitle="Strictly un-fabricated records originating from live model executions">
        {loading ? (
          <TableSkeleton rows={6} cols={6} />
        ) : records.length > 0 ? (
          <div className="space-y-4">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-[#1b2844] text-slate-400">
                    <th className="py-2.5 px-3">TIMESTAMP</th>
                    <th className="py-2.5 px-3">STOCK</th>
                    <th className="py-2.5 px-3">CLOSE PRICE</th>
                    <th className="py-2.5 px-3">P(BULLISH)</th>
                    <th className="py-2.5 px-3">RANK</th>
                    <th className="py-2.5 px-3">RECOMMENDED</th>
                    <th className="py-2.5 px-3">SENTIMENT</th>
                    <th className="py-2.5 px-3">MODEL BUNDLE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#16223b]">
                  {records.map((rec) => (
                    <tr key={rec.id} className="hover:bg-[#141e35] transition-colors">
                      <td className="py-3 px-3 text-slate-400">
                        {new Date(rec.inferenceTimestamp).toLocaleString("en-IN", {
                          timeZone: "Asia/Kolkata",
                        })}
                      </td>
                      <td className="py-3 px-3 font-bold text-white">{rec.stock}</td>
                      <td className="py-3 px-3">₹{rec.closePrice.toFixed(2)}</td>
                      <td className="py-3 px-3 font-bold text-teal-300">
                        {(rec.probability * 100).toFixed(1)}%
                      </td>
                      <td className="py-3 px-3">#{rec.rank}</td>
                      <td className="py-3 px-3">
                        {rec.isRecommended ? (
                          <Badge variant="recommended" size="sm">
                            TOP-K
                          </Badge>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                      <td className="py-3 px-3">
                        <Badge
                          variant={
                            rec.sentimentLabel === "bullish"
                              ? "bullish"
                              : rec.sentimentLabel === "bearish"
                              ? "bearish"
                              : rec.sentimentLabel === "neutral"
                              ? "neutral"
                              : "info"
                          }
                          size="sm"
                        >
                          {rec.sentimentLabel.toUpperCase()}
                        </Badge>
                      </td>
                      <td className="py-3 px-3 text-slate-400 text-[11px] truncate max-w-[140px]">
                        {rec.modelVersion}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination controls */}
            <div className="flex items-center justify-between pt-3 border-t border-[#1b2844] text-xs font-mono">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1 rounded bg-[#0b1020] border border-[#1f2d48] text-slate-300 disabled:opacity-30"
              >
                Previous Page
              </button>
              <span className="text-slate-400">
                Page {page + 1} of {Math.max(1, Math.ceil(total / limit))}
              </span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={(page + 1) * limit >= total}
                className="px-3 py-1 rounded bg-[#0b1020] border border-[#1f2d48] text-slate-300 disabled:opacity-30"
              >
                Next Page
              </button>
            </div>
          </div>
        ) : (
          <div className="py-16 text-center text-xs font-mono text-slate-400 space-y-3">
            <History className="w-8 h-8 text-slate-400 mx-auto" />
            <div className="text-slate-200 font-semibold text-sm">No Genuine Prediction Runs Logged Yet</div>
            <p className="max-w-md mx-auto text-slate-400 leading-relaxed text-xs">
              In accordance with our zero-fabrication policy, no synthetic predictions are seeded into the database.
              Execute an inference run on the <strong>ML Predictions</strong> page to record genuine model outputs here.
            </p>
            <Link
              to="/predictions"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-teal-600 hover:bg-teal-500 text-white font-bold text-xs"
            >
              <span>Go to ML Predictions</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        )}
      </Card>
    </div>
  );
};
