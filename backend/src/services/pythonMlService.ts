import axios, { AxiosInstance } from "axios";
import { env } from "../config/env.js";

export interface PythonServiceHealth {
  status: string;
  service: string;
  model_loaded: boolean;
  model_path: string;
  model_name: string;
  classifier: string;
  features_count: number;
  training_window: string;
  in_sample_auc: number | null;
  supported_stocks_count: number;
  target_universe: string[];
  target_definition: string;
  feed_classification: string;
  timestamp: string;
}

export interface StockItem {
  symbol: string;
  ticker: string;
  name: string;
  sector: string;
  exchange: string;
}

export interface StockCandle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockHistoryResponse {
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
  candles: StockCandle[];
}

export interface PredictionItem {
  stock: string;
  symbol: string;
  close_price: number;
  probability: number;
  rank: number;
  is_recommended: boolean;
  sentiment_label: string;
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
  predictions: PredictionItem[];
  model_metadata: Record<string, any>;
  market_metadata: Record<string, any>;
  news_metadata: Record<string, any>;
}

export interface NewsArticleItem {
  symbol: string;
  title: string;
  source: string;
  url: string;
  published_at: string;
  sentiment_label: string;
  sentiment_score: number;
  is_relevant: boolean;
}

export interface NewsResponse {
  total_articles: number;
  provider: string;
  deduplicated_count: number;
  sentiment_method: string;
  articles: NewsArticleItem[];
}

class PythonMlService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: env.PYTHON_ML_SERVICE_URL,
      timeout: 30000, // 30s timeout for live market/news fetching
      headers: {
        "Content-Type": "application/json",
      },
    });
  }

  async checkHealth(): Promise<{ online: boolean; data?: PythonServiceHealth; error?: string }> {
    try {
      const response = await this.client.get<PythonServiceHealth>("/health", { timeout: 3000 });
      return { online: true, data: response.data };
    } catch (err: any) {
      return {
        online: false,
        error: `Python ML Service offline at ${env.PYTHON_ML_SERVICE_URL} (${err.code || err.message})`,
      };
    }
  }

  async getStocks(): Promise<StockItem[]> {
    const response = await this.client.get<{ count: number; stocks: StockItem[] }>("/api/stocks");
    return response.data.stocks;
  }

  async getStockHistory(symbol: string, period = "6mo"): Promise<StockHistoryResponse> {
    const response = await this.client.get<StockHistoryResponse>(`/api/stocks/${symbol}/history`, {
      params: { period },
    });
    return response.data;
  }

  async runPredict(symbols?: string[], topK = 3, warmupPeriod = "60d"): Promise<PredictResponse> {
    const response = await this.client.post<PredictResponse>("/api/predict", {
      symbols: symbols && symbols.length > 0 ? symbols : undefined,
      top_k: topK,
      warmup_period: warmupPeriod,
    });
    return response.data;
  }

  async getNews(symbols?: string[], limit = 30): Promise<NewsResponse> {
    const symParam = symbols && symbols.length > 0 ? symbols.join(",") : undefined;
    const response = await this.client.get<NewsResponse>("/api/news", {
      params: { symbols: symParam, limit },
    });
    return response.data;
  }
}

export const pythonMlService = new PythonMlService();
