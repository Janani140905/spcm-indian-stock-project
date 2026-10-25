import React, { useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Bar,
  ComposedChart,
} from "recharts";
import { StockHistory } from "../../types/index.js";
import { Skeleton } from "../common/Skeleton.js";

interface StockPriceChartProps {
  history: StockHistory | null;
  loading: boolean;
  onPeriodChange?: (period: string) => void;
  selectedPeriod?: string;
}

const PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"];

export const StockPriceChart: React.FC<StockPriceChartProps> = ({
  history,
  loading,
  onPeriodChange,
  selectedPeriod = "6mo",
}) => {
  const [activePeriod, setActivePeriod] = useState(selectedPeriod);

  const handlePeriod = (p: string) => {
    setActivePeriod(p);
    if (onPeriodChange) onPeriodChange(p);
  };

  if (loading) {
    return (
      <div className="h-72 flex flex-col justify-between p-4">
        <Skeleton className="h-6 w-48 mb-4" />
        <Skeleton className="h-52 w-full" />
      </div>
    );
  }

  if (!history || !history.candles || history.candles.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center border border-dashed border-[#1f2d48] rounded-lg text-slate-400 text-xs">
        No price history series available for charting.
      </div>
    );
  }

  const isPositive = history.change >= 0;
  const strokeColor = isPositive ? "#10b981" : "#ef4444";
  const gradientId = `priceGrad_${history.symbol}`;

  // Calculate min and max for clean Y-axis bounds
  const prices = history.candles.map((c) => c.close);
  const minPrice = Math.floor(Math.min(...prices) * 0.98);
  const maxPrice = Math.ceil(Math.max(...prices) * 1.02);

  return (
    <div className="space-y-4">
      {/* Chart Header & Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#1b2844] pb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-lg font-bold text-white">
              ₹{history.latest_price.toFixed(2)}
            </span>
            <span
              className={`text-xs font-mono font-medium px-2 py-0.5 rounded ${
                isPositive
                  ? "bg-emerald-950/80 text-emerald-400 border border-emerald-500/30"
                  : "bg-rose-950/80 text-rose-400 border border-rose-500/30"
              }`}
            >
              {isPositive ? "+" : ""}
              {history.change.toFixed(2)} ({isPositive ? "+" : ""}
              {history.change_pct.toFixed(2)}%)
            </span>
          </div>
          <p className="text-[11px] text-slate-400 font-mono mt-0.5">
            {history.company_name} ({history.ticker}) · {history.feed_status}
          </p>
        </div>

        {/* Time period selector */}
        <div className="flex items-center gap-1 bg-[#0c1322] p-1 rounded-lg border border-[#1b2844]">
          {PERIODS.map((period) => (
            <button
              key={period}
              onClick={() => handlePeriod(period)}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                activePeriod === period
                  ? "bg-violet-600/80 text-white font-semibold shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {period.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Recharts Area Chart */}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={history.candles}
            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
          >
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={strokeColor} stopOpacity={0.4} />
                <stop offset="95%" stopColor={strokeColor} stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={{ stroke: "#1f2d48" }}
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
              tickFormatter={(d: string) => {
                const parts = d.split("-");
                return parts.length >= 3 ? `${parts[1]}/${parts[2]}` : d;
              }}
              minTickGap={30}
            />

            {/* Price Y-Axis */}
            <YAxis
              yAxisId="price"
              domain={[minPrice, maxPrice]}
              tickLine={false}
              axisLine={{ stroke: "#1f2d48" }}
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
              tickFormatter={(v: number) => `₹${v}`}
            />

            {/* Volume Y-Axis (background histogram) */}
            <YAxis
              yAxisId="volume"
              orientation="right"
              domain={[0, (dataMax: number) => dataMax * 4]}
              hide
            />

            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const data = payload[0].payload;
                return (
                  <div className="bg-[#0b1020]/95 border border-[#2a3c5e] p-3 rounded-lg shadow-xl font-mono text-xs space-y-1">
                    <div className="text-slate-400 border-b border-slate-800 pb-1 mb-1">
                      {data.date}
                    </div>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-400">Close:</span>
                      <span className="font-bold text-white">₹{data.close.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between gap-4 text-[10px]">
                      <span className="text-slate-500">O / H / L:</span>
                      <span className="text-slate-300">
                        ₹{data.open} / ₹{data.high} / ₹{data.low}
                      </span>
                    </div>
                    <div className="flex justify-between gap-4 text-[10px]">
                      <span className="text-slate-500">Volume:</span>
                      <span className="text-slate-400">
                        {data.volume ? data.volume.toLocaleString() : "N/A"}
                      </span>
                    </div>
                  </div>
                );
              }}
            />

            <Area
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke={strokeColor}
              strokeWidth={2}
              fillOpacity={1}
              fill={`url(#${gradientId})`}
            />

            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="#334155"
              opacity={0.3}
              radius={[2, 2, 0, 0]}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
