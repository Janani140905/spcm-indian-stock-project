import {
  Stock,
  StockHistory,
  PredictResponse,
  NewsResponse,
  WatchlistItem,
  PredictionHistoryRecord,
  SystemHealth,
} from "../types/index.js";

const API_BASE = "/api";

// Fallback demo data - clearly isolated per Real-data and Demo-data policy
export const DEMO_PREDICTIONS: PredictResponse = {
  status: "success",
  is_demo_data: true,
  inference_timestamp: new Date().toISOString(),
  top_k: 3,
  recommended_stocks: ["SBIN", "RELIANCE", "KOTAKBANK"],
  predictions: [
    {
      stock: "SBIN",
      symbol: "SBIN.NS",
      close_price: 965.0,
      probability: 0.3096,
      rank: 1,
      is_recommended: true,
      sentiment_label: "no_news",
      news_count: 0,
      sent_score: null,
      pv_volume: null,
      ai_index: null,
      observation_timestamp: new Date().toISOString(),
      trading_date: "2026-10-21",
    },
    {
      stock: "RELIANCE",
      symbol: "RELIANCE.NS",
      close_price: 1206.5,
      probability: 0.2804,
      rank: 2,
      is_recommended: true,
      sentiment_label: "neutral",
      news_count: 1,
      sent_score: null,
      pv_volume: null,
      ai_index: null,
      observation_timestamp: new Date().toISOString(),
      trading_date: "2026-10-21",
    },
    {
      stock: "KOTAKBANK",
      symbol: "KOTAKBANK.NS",
      close_price: 436.95,
      probability: 0.2556,
      rank: 3,
      is_recommended: true,
      sentiment_label: "neutral",
      news_count: 1,
      sent_score: null,
      pv_volume: null,
      ai_index: null,
      observation_timestamp: new Date().toISOString(),
      trading_date: "2026-10-21",
    },
    {
      stock: "ITC",
      symbol: "ITC.NS",
      close_price: 261.1,
      probability: 0.248,
      rank: 4,
      is_recommended: false,
      sentiment_label: "neutral",
      news_count: 1,
      sent_score: null,
      pv_volume: null,
      ai_index: null,
      observation_timestamp: new Date().toISOString(),
      trading_date: "2026-10-21",
    },
    {
      stock: "TCS",
      symbol: "TCS.NS",
      close_price: 2515.9,
      probability: 0.2474,
      rank: 5,
      is_recommended: false,
      sentiment_label: "neutral",
      news_count: 2,
      sent_score: null,
      pv_volume: null,
      ai_index: null,
      observation_timestamp: new Date().toISOString(),
      trading_date: "2026-10-21",
    },
    {
      stock: "INFY",
      symbol: "INFY.NS",
      close_price: 1165.0,
      probability: 0.202,
      rank: 6,
      is_recommended: false,
      sentiment_label: "bullish",
      news_count: 4,
      sent_score: 1.0,
      pv_volume: 0.6931,
      ai_index: 1.0,
      observation_timestamp: new Date().toISOString(),
      trading_date: "2026-10-21",
    },
  ],
  model_metadata: {
    model_name: "spcm_nn_price_and_sentiment (DEMO)",
    classifier: "nn",
    feature_count: 45,
    training_samples: 9760,
    in_sample_auc: 0.5183,
  },
  market_metadata: {
    provider: "yfinance (Demo Mode)",
    feed_type: "DEMO_ILLUSTRATIVE",
    delay_minutes: 15,
  },
  news_metadata: {
    provider: "google_news (Demo Mode)",
    sentiment_method: "finbert_or_lexicon",
  },
};

export async function fetchHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok && res.status !== 207) {
    throw new Error(`Failed to fetch health status (${res.status})`);
  }
  return res.json();
}

export async function fetchStocks(): Promise<Stock[]> {
  const res = await fetch(`${API_BASE}/stocks`);
  if (!res.ok) throw new Error("Failed to fetch stock catalog");
  const data = await res.json();
  return data.stocks;
}

export async function fetchStockHistory(symbol: string, period = "6mo"): Promise<StockHistory> {
  const res = await fetch(`${API_BASE}/stocks/${symbol}/history?period=${period}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Failed to fetch history for ${symbol}`);
  }
  const json = await res.json();
  return json.data;
}

export async function runModelPrediction(
  symbols?: string[],
  topK = 3,
  warmupPeriod = "60d"
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predictions/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbols, top_k: topK, warmup_period: warmupPeriod }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || "Inference execution failed");
  }
  const json = await res.json();
  return json.data;
}

export async function fetchNews(symbols?: string[], limit = 30): Promise<NewsResponse> {
  const query = new URLSearchParams();
  if (symbols && symbols.length > 0) query.set("symbols", symbols.join(","));
  query.set("limit", String(limit));

  const res = await fetch(`${API_BASE}/news?${query.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch news stream");
  const json = await res.json();
  return json.data;
}

export async function fetchWatchlist(): Promise<WatchlistItem[]> {
  const res = await fetch(`${API_BASE}/watchlist`);
  if (!res.ok) throw new Error("Failed to fetch watchlist");
  const json = await res.json();
  return json.items;
}

export async function addToWatchlist(item: {
  symbol: string;
  targetPrice?: number | null;
  notes?: string | null;
}): Promise<WatchlistItem> {
  const res = await fetch(`${API_BASE}/watchlist`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(item),
  });
  if (!res.ok) throw new Error("Failed to add to watchlist");
  const json = await res.json();
  return json.item;
}

export async function removeFromWatchlist(symbol: string): Promise<void> {
  const res = await fetch(`${API_BASE}/watchlist/${symbol}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`Failed to remove ${symbol} from watchlist`);
}

export async function fetchPredictionHistory(options?: {
  stock?: string;
  limit?: number;
  offset?: number;
}): Promise<{ total: number; records: PredictionHistoryRecord[]; source: string }> {
  const query = new URLSearchParams();
  if (options?.stock) query.set("stock", options.stock);
  if (options?.limit) query.set("limit", String(options.limit));
  if (options?.offset) query.set("offset", String(options.offset));

  const res = await fetch(`${API_BASE}/predictions/history?${query.toString()}`);
  if (!res.ok) throw new Error("Failed to fetch prediction history");
  return res.json();
}
