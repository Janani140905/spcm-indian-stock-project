import axios from "axios";
import {
  STOCK_UNIVERSE,
  STOCK_BY_SYMBOL,
  SECTORS,
  StockDefinition,
  SectorDefinition,
  CANONICAL_BENCHMARK_SYMBOLS,
} from "../config/stockUniverse.js";

export interface StockQuote {
  symbol: string;
  ticker: string;
  name: string;
  sector: string;
  exchange: "NSE";
  currentPrice: number;
  previousClose: number;
  change: number;
  changePct: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  fiftyTwoWeekHigh: number | null;
  fiftyTwoWeekLow: number | null;
  timestamp: string;
  marketTimeIST: string;
  feedStatus: "DELAYED_15M" | "HISTORICAL_EOD";
  inTrainingUniverse: boolean;
  marketCapCategory: "Large Cap" | "Mid Cap";
  description: string;
  technicalIndicators: {
    sma20: number | null;
    sma50: number | null;
    volatility30d: number | null;
    momentum10d: number | null;
    rsi14: number | null;
  };
}

export interface CandleData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface StockHistoryResult {
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
  timestamp: string;
  in_training_universe: boolean;
  candles: CandleData[];
  indicators: {
    sma20: number | null;
    sma50: number | null;
    volatility30d: number | null;
    momentum10d: number | null;
  };
}

interface CacheItem<T> {
  data: T;
  timestamp: number;
}

class MarketDataService {
  private quoteCache: Map<string, CacheItem<StockQuote>> = new Map();
  private historyCache: Map<string, CacheItem<StockHistoryResult>> = new Map();
  private CACHE_TTL_MS = 5 * 60 * 1000; // 5-minute in-memory cache to prevent Yahoo Finance throttling

  /**
   * Helper to format Date into Indian Standard Time (IST)
   */
  private formatIST(date: Date): string {
    return date.toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
      dateStyle: "medium",
      timeStyle: "medium",
    }) + " IST";
  }

  /**
   * Determine feed classification (Delayed during Indian trading hours, Historical outside)
   */
  private getFeedStatus(marketDate: Date): "DELAYED_15M" | "HISTORICAL_EOD" {
    // Convert to IST
    const istHours = (marketDate.getUTCHours() + 5 + Math.floor((marketDate.getUTCMinutes() + 30) / 60)) % 24;
    const istMinutes = (marketDate.getUTCMinutes() + 30) % 60;
    const dayOfWeek = marketDate.getUTCDay();

    // NSE trading hours: Mon-Fri 09:15 to 15:30 IST
    const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
    const timeInMinutes = istHours * 60 + istMinutes;
    const isMarketHours = isWeekday && timeInMinutes >= 9 * 60 + 15 && timeInMinutes <= 15 * 60 + 30;

    return isMarketHours ? "DELAYED_15M" : "HISTORICAL_EOD";
  }

  /**
   * Get all stocks, optionally filtered by sector or text query
   */
  getStocks(sector?: string, query?: string, benchmarkOnly?: boolean): StockDefinition[] {
    let result = STOCK_UNIVERSE;

    if (benchmarkOnly) {
      const benchSet = new Set(CANONICAL_BENCHMARK_SYMBOLS as readonly string[]);
      result = result.filter((s) => benchSet.has(s.symbol));
    }

    if (sector && sector !== "All") {
      const lower = sector.toLowerCase();
      result = result.filter((s) => s.sector.toLowerCase() === lower);
    }

    if (query && query.trim()) {
      const q = query.trim().toLowerCase();
      result = result.filter(
        (s) =>
          s.symbol.toLowerCase().includes(q) ||
          s.name.toLowerCase().includes(q) ||
          s.sector.toLowerCase().includes(q)
      );
    }

    return result;
  }

  /**
   * Get all 12 sectors with metadata
   */
  getSectors(): SectorDefinition[] {
    return SECTORS;
  }

  /**
   * Fetch live/delayed stock quote with technical metrics from Yahoo Finance
   */
  async getStockQuote(symbol: string): Promise<StockQuote> {
    const cleanSym = symbol.toUpperCase().replace(".NS", "");
    const def = STOCK_BY_SYMBOL[cleanSym];
    if (!def) {
      throw new Error(`Stock '${symbol}' is not part of the supported Indian universe.`);
    }

    // Check cache
    const cached = this.quoteCache.get(cleanSym);
    const now = Date.now();
    if (cached && now - cached.timestamp < this.CACHE_TTL_MS) {
      return cached.data;
    }

    const ticker = def.ticker;
    const queryTicker = cleanSym === "TATAMOTORS" ? "TMPV.NS" : def.ticker;
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${queryTicker}?range=60d&interval=1d`;

    try {
      const res = await axios.get(url, {
        headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
        timeout: 10000,
      });

      const chart = res.data?.chart?.result?.[0];
      if (!chart) {
        throw new Error(`No market data returned from provider for ${ticker}`);
      }

      const meta = chart.meta || {};
      const timestamps: number[] = chart.timestamp || [];
      const quote = chart.indicators?.quote?.[0] || {};
      const closes: (number | null)[] = quote.close || [];
      const opens: (number | null)[] = quote.open || [];
      const highs: (number | null)[] = quote.high || [];
      const lows: (number | null)[] = quote.low || [];
      const volumes: (number | null)[] = quote.volume || [];

      // Valid closing prices
      const validCloses: number[] = closes.filter((c): c is number => c !== null && !isNaN(c));

      const currentPrice =
        meta.regularMarketPrice ??
        (validCloses.length > 0 ? validCloses[validCloses.length - 1] : 0);
      const previousClose =
        meta.chartPreviousClose ??
        (validCloses.length >= 2 ? validCloses[validCloses.length - 2] : currentPrice);

      const change = round(currentPrice - previousClose, 2);
      const changePct = previousClose > 0 ? round((change / previousClose) * 100, 2) : 0;

      const lastIdx = closes.length - 1;
      const open = meta.regularMarketOpen ?? (lastIdx >= 0 && opens[lastIdx] != null ? opens[lastIdx]! : currentPrice);
      const high = meta.regularMarketDayHigh ?? (lastIdx >= 0 && highs[lastIdx] != null ? highs[lastIdx]! : currentPrice);
      const low = meta.regularMarketDayLow ?? (lastIdx >= 0 && lows[lastIdx] != null ? lows[lastIdx]! : currentPrice);
      const volume = meta.regularMarketVolume ?? (lastIdx >= 0 && volumes[lastIdx] != null ? volumes[lastIdx]! : 0);

      // Technical metrics calculation
      const sma20 = calculateSMA(validCloses, 20);
      const sma50 = calculateSMA(validCloses, 50);
      const volatility30d = calculateAnnualizedVolatility(validCloses, 30);
      const momentum10d = calculateMomentum(validCloses, 10);
      const rsi14 = calculateRSI(validCloses, 14);

      const marketDate = meta.regularMarketTime
        ? new Date(meta.regularMarketTime * 1000)
        : new Date();

      const quoteResult: StockQuote = {
        symbol: cleanSym,
        ticker,
        name: def.name,
        sector: def.sector,
        exchange: def.exchange,
        currentPrice: round(currentPrice, 2),
        previousClose: round(previousClose, 2),
        change,
        changePct,
        open: round(open, 2),
        high: round(high, 2),
        low: round(low, 2),
        volume: Math.round(volume),
        fiftyTwoWeekHigh: meta.fiftyTwoWeekHigh ? round(meta.fiftyTwoWeekHigh, 2) : null,
        fiftyTwoWeekLow: meta.fiftyTwoWeekLow ? round(meta.fiftyTwoWeekLow, 2) : null,
        timestamp: marketDate.toISOString(),
        marketTimeIST: this.formatIST(marketDate),
        feedStatus: this.getFeedStatus(marketDate),
        inTrainingUniverse: def.inTrainingUniverse,
        marketCapCategory: def.marketCapCategory,
        description: def.description,
        technicalIndicators: {
          sma20,
          sma50,
          volatility30d,
          momentum10d,
          rsi14,
        },
      };

      this.quoteCache.set(cleanSym, { data: quoteResult, timestamp: now });
      return quoteResult;
    } catch (err: any) {
      // If error but cache exists, return stale cache
      if (cached) {
        console.warn(`[MarketDataService] Provider fetch failed for ${ticker}; returning stale cache:`, err.message);
        return cached.data;
      }
      throw new Error(`Failed to fetch live market quote for ${symbol}: ${err.message}`);
    }
  }

  /**
   * Fetch historical OHLCV candles
   */
  async getStockHistory(symbol: string, period = "6mo"): Promise<StockHistoryResult> {
    const cleanSym = symbol.toUpperCase().replace(".NS", "");
    const def = STOCK_BY_SYMBOL[cleanSym];
    if (!def) {
      throw new Error(`Stock '${symbol}' is not part of the supported Indian universe.`);
    }

    const cacheKey = `${cleanSym}_${period}`;
    const cached = this.historyCache.get(cacheKey);
    const now = Date.now();
    if (cached && now - cached.timestamp < this.CACHE_TTL_MS) {
      return cached.data;
    }

    const ticker = def.ticker;
    const queryTicker = cleanSym === "TATAMOTORS" ? "TMPV.NS" : def.ticker;
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${queryTicker}?range=${period}&interval=1d`;

    try {
      const res = await axios.get(url, {
        headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
        timeout: 12000,
      });

      const chart = res.data?.chart?.result?.[0];
      if (!chart) {
        throw new Error(`No historical series found for ${ticker}`);
      }

      const timestamps: number[] = chart.timestamp || [];
      const quote = chart.indicators?.quote?.[0] || {};
      const opens: (number | null)[] = quote.open || [];
      const highs: (number | null)[] = quote.high || [];
      const lows: (number | null)[] = quote.low || [];
      const closes: (number | null)[] = quote.close || [];
      const volumes: (number | null)[] = quote.volume || [];

      const candles: CandleData[] = [];
      const validCloses: number[] = [];

      for (let i = 0; i < timestamps.length; i++) {
        const c = closes[i];
        if (c === null || isNaN(c)) continue;

        const dt = new Date(timestamps[i] * 1000);
        const dateStr = dt.toISOString().split("T")[0];

        candles.push({
          date: dateStr,
          open: round(opens[i] ?? c, 2),
          high: round(highs[i] ?? c, 2),
          low: round(lows[i] ?? c, 2),
          close: round(c, 2),
          volume: Math.round(volumes[i] ?? 0),
        });
        validCloses.push(c);
      }

      const latestPrice = validCloses.length > 0 ? validCloses[validCloses.length - 1] : 0;
      const prevPrice = validCloses.length >= 2 ? validCloses[validCloses.length - 2] : latestPrice;
      const change = round(latestPrice - prevPrice, 2);
      const changePct = prevPrice > 0 ? round((change / prevPrice) * 100, 2) : 0;

      const historyResult: StockHistoryResult = {
        symbol: cleanSym,
        ticker,
        company_name: def.name,
        sector: def.sector,
        period,
        record_count: candles.length,
        latest_price: round(latestPrice, 2),
        change,
        change_pct: changePct,
        feed_status: "HISTORICAL_EOD",
        timestamp: new Date().toISOString(),
        in_training_universe: def.inTrainingUniverse,
        candles,
        indicators: {
          sma20: calculateSMA(validCloses, 20),
          sma50: calculateSMA(validCloses, 50),
          volatility30d: calculateAnnualizedVolatility(validCloses, 30),
          momentum10d: calculateMomentum(validCloses, 10),
        },
      };

      this.historyCache.set(cacheKey, { data: historyResult, timestamp: now });
      return historyResult;
    } catch (err: any) {
      if (cached) {
        console.warn(`[MarketDataService] Provider history failed for ${ticker}; returning stale cache:`, err.message);
        return cached.data;
      }
      throw new Error(`Failed to retrieve history for ${symbol}: ${err.message}`);
    }
  }
}

// Math and Financial Indicator Utilities
function round(val: number, decimals = 2): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

function calculateSMA(series: number[], window: number): number | null {
  if (series.length < window) return null;
  const slice = series.slice(-window);
  const sum = slice.reduce((acc, v) => acc + v, 0);
  return round(sum / window, 2);
}

function calculateMomentum(series: number[], days = 10): number | null {
  if (series.length <= days) return null;
  const current = series[series.length - 1];
  const past = series[series.length - 1 - days];
  if (past === 0) return 0;
  return round(((current - past) / past) * 100, 2);
}

function calculateAnnualizedVolatility(series: number[], days = 30): number | null {
  if (series.length <= days) return null;
  const slice = series.slice(-days - 1);
  const returns: number[] = [];
  for (let i = 1; i < slice.length; i++) {
    const prev = slice[i - 1];
    if (prev > 0) {
      returns.push((slice[i] - prev) / prev);
    }
  }
  if (returns.length < 2) return null;

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((acc, r) => acc + Math.pow(r - mean, 2), 0) / (returns.length - 1);
  const dailyStdDev = Math.sqrt(variance);
  const annualized = dailyStdDev * Math.sqrt(252) * 100;
  return round(annualized, 2);
}

function calculateRSI(series: number[], period = 14): number | null {
  if (series.length <= period) return null;
  let gains = 0;
  let losses = 0;

  for (let i = 1; i <= period; i++) {
    const diff = series[i] - series[i - 1];
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }

  let avgGain = gains / period;
  let avgLoss = losses / period;

  for (let i = period + 1; i < series.length; i++) {
    const diff = series[i] - series[i - 1];
    if (diff >= 0) {
      avgGain = (avgGain * (period - 1) + diff) / period;
      avgLoss = (avgLoss * (period - 1)) / period;
    } else {
      avgGain = (avgGain * (period - 1)) / period;
      avgLoss = (avgLoss * (period - 1) - diff) / period;
    }
  }

  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return round(100 - 100 / (1 + rs), 2);
}

export const marketDataService = new MarketDataService();
