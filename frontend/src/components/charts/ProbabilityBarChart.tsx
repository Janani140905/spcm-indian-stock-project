import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ReferenceLine,
} from "recharts";
import { Prediction } from "../../types/index.js";

interface ProbabilityBarChartProps {
  predictions: Prediction[];
}

export const ProbabilityBarChart: React.FC<ProbabilityBarChartProps> = ({ predictions }) => {
  if (!predictions || predictions.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-400 border border-dashed border-[#1f2d48] rounded-lg">
        No active prediction probabilities available.
      </div>
    );
  }

  // Sort descending by probability
  const chartData = [...predictions]
    .sort((a, b) => b.probability - a.probability)
    .map((p) => ({
      stock: p.stock,
      probability: Number((p.probability * 100).toFixed(1)),
      rawProb: p.probability,
      isRecommended: p.is_recommended,
      rank: p.rank,
      sentiment: p.sentiment_label,
    }));

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs font-mono text-slate-400 px-1">
        <span>Cross-Sectional Bullish Probability P(y=1)</span>
        <span className="text-teal-400 font-semibold">■ Top-K Recommended Stocks</span>
      </div>

      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={chartData}
            margin={{ top: 5, right: 25, left: 10, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 45]}
              unit="%"
              tickLine={false}
              axisLine={{ stroke: "#1f2d48" }}
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "monospace" }}
            />
            <YAxis
              type="category"
              dataKey="stock"
              tickLine={false}
              axisLine={{ stroke: "#1f2d48" }}
              tick={{ fill: "#cbd5e1", fontSize: 11, fontFamily: "monospace", fontWeight: 600 }}
              width={85}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const item = payload[0].payload;
                return (
                  <div className="bg-[#0b1020]/95 border border-[#2a3c5e] p-2.5 rounded-lg font-mono text-xs shadow-xl space-y-1">
                    <div className="text-white font-bold flex items-center justify-between gap-4">
                      <span>{item.stock}</span>
                      <span className="text-[10px] text-teal-300">Rank #{item.rank}</span>
                    </div>
                    <div className="text-slate-300">
                      P(Bullish): <span className="font-bold text-teal-400">{item.probability}%</span>
                    </div>
                    <div className="text-[10px] text-slate-400">
                      Status: {item.isRecommended ? "Recommended in Top-K" : "Below Top-K Cutoff"}
                    </div>
                    <div className="text-[10px] text-slate-400">
                      Sentiment: {item.sentiment.toUpperCase()}
                    </div>
                  </div>
                );
              }}
            />
            <ReferenceLine
              x={25}
              stroke="#6366f1"
              strokeDasharray="3 3"
              label={{
                value: "Baseline ~25%",
                fill: "#818cf8",
                fontSize: 9,
                position: "insideTopRight",
              }}
            />
            <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isRecommended ? "#14b8a6" : "#334155"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
