export interface Stock {
  symbol: string;
  ticker: string;
  name: string;
  sector: string;
  exchange: string;
}

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockHistory {
  symbol: string;
  ticker: string;
  company_name: string;
  sector: string;
  period: string;
  record_count: number;
  latest_price: number;
  change: number;
  change_pct: number;
  feed_status: string;
  candles: Candle[];
}

export interface Prediction {
  stock: string;
  symbol: string;
  close_price: number;
  probability: number;
  rank: number;
  is_recommended: boolean;
  sentiment_label: "bullish" | "bearish" | "neutral" | "no_news" | string;
  news_count: number;
  sent_score: number | null;
  pv_volume: number | null;
  ai_index: number | null;
  observation_timestamp: string;
  trading_date: string;
}

export interface PredictResponse {
  status: string;
  is_demo_data: boolean;
  inference_timestamp: string;
  top_k: number;
  recommended_stocks: string[];
  predictions: Prediction[];
  model_metadata: {
    model_name?: string;
    classifier?: string;
    feature_count?: number;
    start_date?: string;
    end_date?: string;
    training_samples?: number;
    in_sample_auc?: number;
    target_stocks?: string[];
  };
  market_metadata: {
    provider?: string;
    feed_type?: string;
    delay_minutes?: number;
    is_live?: boolean;
    market_session?: string;
    symbols_requested?: string[];
    symbols_successful?: string[];
    symbols_failed?: Record<string, string>;
    latest_timestamp?: string | null;
    symbols_fetched?: string[];
    total_rows?: number;
  };
  news_metadata: {
    provider?: string;
    symbols_requested?: string[];
    symbols_successful?: string[];
    symbols_failed?: Record<string, string>;
    total_articles_fetched?: number;
    total_articles_deduplicated?: number;
    rate_limit_info?: string;
    notes?: string;
    sentiment_method?: string;
  };
}

export interface NewsArticle {
  symbol: string;
  title: string;
  source: string;
  url: string;
  published_at: string;
  sentiment_label: "bullish" | "bearish" | "neutral" | string;
  sentiment_score: number;
  is_relevant: boolean;
}

export interface NewsResponse {
  total_articles: number;
  provider: string;
  deduplicated_count: number;
  sentiment_method: string;
  articles: NewsArticle[];
}

export interface WatchlistItem {
  id: string;
  watchlistId: string;
  symbol: string;
  ticker: string;
  targetPrice: number | null;
  notes: string | null;
  addedAt: string;
}

export interface PredictionHistoryRecord {
  id: string;
  stock: string;
  symbol: string;
  closePrice: number;
  probability: number;
  rank: number;
  isRecommended: boolean;
  sentimentLabel: string;
  sentScore: number | null;
  pvVolume: number | null;
  aiIndex: number | null;
  newsCount: number;
  modelVersion: string;
  marketDate: string;
  observationTimestamp: string;
  inferenceTimestamp: string;
  isDemoData: boolean;
}

export interface SystemHealth {
  status: "healthy" | "degraded" | "down";
  server: string;
  timestamp: string;
  services: {
    express_api: {
      status: "up" | "down";
      port: number;
      env: string;
    };
    python_ml_engine: {
      status: "up" | "down";
      url: string;
      details?: {
        model_loaded: boolean;
        model_name: string;
        classifier: string;
        features_count: number;
        training_window: string;
        in_sample_auc: number;
        supported_stocks_count: number;
        target_universe: string[];
        feed_classification: string;
      } | null;
      error?: string | null;
    };
    database: {
      status: "connected" | "in_memory_fallback";
      provider: string;
      mode: "live" | "in_memory_fallback";
      error?: string | null;
    };
  };
  canonical_universe: string[];
}
