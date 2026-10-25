import { PrismaClient } from "@prisma/client";
import { env } from "../config/env.js";

export interface DatabaseStatus {
  isConnected: boolean;
  provider: "postgresql";
  mode: "live" | "in_memory_fallback";
  error?: string;
}

export interface WatchlistRecord {
  id: string;
  watchlistId: string;
  symbol: string;
  ticker: string;
  targetPrice: number | null;
  notes: string | null;
  addedAt: Date;
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
  observationTimestamp: Date;
  inferenceTimestamp: Date;
  isDemoData: boolean;
}

// In-memory fallback repository when PostgreSQL is offline/unreachable
class InMemoryStore {
  private watchlist: Map<string, WatchlistRecord> = new Map();
  private predictionHistory: PredictionHistoryRecord[] = [];

  constructor() {
    // Seed default watchlist with 3 core NSE stocks
    const now = new Date();
    this.addWatchlistItem({
      symbol: "RELIANCE",
      ticker: "RELIANCE.NS",
      targetPrice: 1300.0,
      notes: "Heavyweight index constituent",
    });
    this.addWatchlistItem({
      symbol: "TCS",
      ticker: "TCS.NS",
      targetPrice: 2600.0,
      notes: "IT sector benchmark",
    });
  }

  getWatchlist(): WatchlistRecord[] {
    return Array.from(this.watchlist.values()).sort(
      (a, b) => b.addedAt.getTime() - a.addedAt.getTime()
    );
  }

  addWatchlistItem(item: {
    symbol: string;
    ticker: string;
    targetPrice?: number | null;
    notes?: string | null;
  }): WatchlistRecord {
    const existing = this.watchlist.get(item.symbol.toUpperCase());
    if (existing) {
      existing.targetPrice = item.targetPrice ?? existing.targetPrice;
      existing.notes = item.notes ?? existing.notes;
      return existing;
    }
    const record: WatchlistRecord = {
      id: "mem_" + Math.random().toString(36).substring(2, 11),
      watchlistId: "default_watchlist",
      symbol: item.symbol.toUpperCase(),
      ticker: item.ticker.toUpperCase(),
      targetPrice: item.targetPrice ?? null,
      notes: item.notes ?? null,
      addedAt: new Date(),
    };
    this.watchlist.set(record.symbol, record);
    return record;
  }

  removeWatchlistItem(symbol: string): boolean {
    return this.watchlist.delete(symbol.toUpperCase());
  }

  recordPredictions(records: Omit<PredictionHistoryRecord, "id" | "inferenceTimestamp">[]): PredictionHistoryRecord[] {
    const created: PredictionHistoryRecord[] = [];
    for (const r of records) {
      const rec: PredictionHistoryRecord = {
        ...r,
        id: "pred_" + Math.random().toString(36).substring(2, 11),
        inferenceTimestamp: new Date(),
      };
      this.predictionHistory.unshift(rec);
      created.push(rec);
    }
    return created;
  }

  getPredictionHistory(options?: {
    stock?: string;
    limit?: number;
    offset?: number;
  }): { total: number; records: PredictionHistoryRecord[] } {
    let list = this.predictionHistory;
    if (options?.stock) {
      const s = options.stock.toUpperCase();
      list = list.filter((r) => r.stock === s || r.symbol === s);
    }
    const total = list.length;
    const offset = options?.offset ?? 0;
    const limit = options?.limit ?? 50;
    const records = list.slice(offset, offset + limit);
    return { total, records };
  }
}

export const inMemoryStore = new InMemoryStore();

let prismaInstance: PrismaClient | null = null;
export let dbStatus: DatabaseStatus = {
  isConnected: false,
  provider: "postgresql",
  mode: "in_memory_fallback",
  error: "Database initialization pending",
};

export async function initDatabase(): Promise<DatabaseStatus> {
  try {
    const client = new PrismaClient({
      datasources: {
        db: {
          url: env.DATABASE_URL,
        },
      },
      log: ["warn", "error"],
    });

    // Test connection with a timeout
    await Promise.race([
      client.$connect(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("PostgreSQL connection timeout (3000ms)")), 3000)
      ),
    ]);

    prismaInstance = client;
    dbStatus = {
      isConnected: true,
      provider: "postgresql",
      mode: "live",
    };
    console.log("[Database] Connected successfully to PostgreSQL via Prisma.");
    return dbStatus;
  } catch (err: any) {
    const errMsg = err?.message || String(err);
    dbStatus = {
      isConnected: false,
      provider: "postgresql",
      mode: "in_memory_fallback",
      error: `PostgreSQL connection failed (${errMsg}). Operating in local in-memory persistence mode.`,
    };
    console.warn(
      `[Database] Note: PostgreSQL is not reachable at ${env.DATABASE_URL}. ` +
      `Express is operating in local in-memory persistence mode. ` +
      `To connect PostgreSQL, install PostgreSQL, create database 'spcm_intelligence', and run 'npx prisma migrate dev'.`
    );
    return dbStatus;
  }
}

export function getPrismaClient(): PrismaClient | null {
  return prismaInstance;
}
