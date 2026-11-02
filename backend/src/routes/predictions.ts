import { Router, Request, Response } from "express";
import { z } from "zod";
import { pythonMlService, PredictionItem } from "../services/pythonMlService.js";
import { getPrismaClient, inMemoryStore, dbStatus } from "../db/prisma.js";
import { STOCK_UNIVERSE } from "../config/stockUniverse.js";

export const predictionsRouter = Router();

const predictRequestSchema = z.object({
  symbols: z.array(z.string()).optional(),
  top_k: z.number().int().min(1).max(50).default(3),
  warmup_period: z.string().default("60d"),
});

const personalizedRequestSchema = z.object({
  profile: z
    .object({
      riskTolerance: z.enum(["conservative", "moderate", "aggressive"]).default("moderate"),
      preferredSectors: z
        .array(z.string())
        .default(["Information Technology", "Banking and Financial Services", "Energy and Oil & Gas"]),
      investmentHorizon: z.enum(["short_term", "medium_term", "long_term"]).default("medium_term"),
      sentimentPreference: z.enum(["high_sentiment", "balanced", "pure_price"]).default("balanced"),
    })
    .default({
      riskTolerance: "moderate",
      preferredSectors: ["Information Technology", "Banking and Financial Services", "Energy and Oil & Gas"],
      investmentHorizon: "medium_term",
      sentimentPreference: "balanced",
    }),
  symbols: z.array(z.string()).optional(),
  top_k: z.number().int().min(1).max(50).default(3),
});

const STOCK_SECTORS: Record<string, string> = STOCK_UNIVERSE.reduce((acc, s) => {
  acc[s.symbol.toUpperCase()] = s.sector;
  return acc;
}, {} as Record<string, string>);

const DEMO_FALLBACK_PREDICTIONS: PredictionItem[] = [
  { stock: "SBIN", symbol: "SBIN.NS", close_price: 965.0, probability: 0.3096, rank: 1, is_recommended: true, sentiment_label: "no_news", news_count: 0, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "RELIANCE", symbol: "RELIANCE.NS", close_price: 1206.5, probability: 0.2804, rank: 2, is_recommended: true, sentiment_label: "neutral", news_count: 1, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "KOTAKBANK", symbol: "KOTAKBANK.NS", close_price: 436.95, probability: 0.2556, rank: 3, is_recommended: true, sentiment_label: "neutral", news_count: 1, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "ITC", symbol: "ITC.NS", close_price: 261.1, probability: 0.248, rank: 4, is_recommended: false, sentiment_label: "neutral", news_count: 1, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "TCS", symbol: "TCS.NS", close_price: 2515.9, probability: 0.2474, rank: 5, is_recommended: false, sentiment_label: "neutral", news_count: 2, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "INFY", symbol: "INFY.NS", close_price: 1165.0, probability: 0.202, rank: 6, is_recommended: false, sentiment_label: "bullish", news_count: 4, sent_score: 1.0, pv_volume: 0.6931, ai_index: 1.0, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "HDFCBANK", symbol: "HDFCBANK.NS", close_price: 1510.0, probability: 0.198, rank: 7, is_recommended: false, sentiment_label: "neutral", news_count: 1, sent_score: 0.0, pv_volume: 0.2, ai_index: 0.5, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "ICICIBANK", symbol: "ICICIBANK.NS", close_price: 1120.0, probability: 0.195, rank: 8, is_recommended: false, sentiment_label: "neutral", news_count: 0, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "HINDUNILVR", symbol: "HINDUNILVR.NS", close_price: 2450.0, probability: 0.189, rank: 9, is_recommended: false, sentiment_label: "neutral", news_count: 1, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
  { stock: "BHARTIARTL", symbol: "BHARTIARTL.NS", close_price: 1680.0, probability: 0.180, rank: 10, is_recommended: false, sentiment_label: "neutral", news_count: 0, sent_score: null, pv_volume: null, ai_index: null, observation_timestamp: new Date().toISOString(), trading_date: "2026-10-21" },
];

// POST /api/predictions/run - Execute genuine inference and persist
predictionsRouter.post("/run", async (req: Request, res: Response) => {
  const parseResult = predictRequestSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      success: false,
      error: "Invalid request parameters",
      details: parseResult.error.format(),
    });
  }

  const { symbols, top_k, warmup_period } = parseResult.data;

  try {
    // 1. Call Python ML engine
    const predictionResult = await pythonMlService.runPredict(symbols, top_k, warmup_period);

    // 2. Persist genuine predictions to Database / store
    const prisma = getPrismaClient();
    const recordsToSave = predictionResult.predictions.map((p) => ({
      stock: p.stock,
      symbol: p.symbol,
      closePrice: p.close_price,
      probability: p.probability,
      rank: p.rank,
      isRecommended: p.is_recommended,
      sentimentLabel: p.sentiment_label,
      sentScore: p.sent_score,
      pvVolume: p.pv_volume,
      aiIndex: p.ai_index,
      newsCount: p.news_count,
      modelVersion: (predictionResult.model_metadata.model_name as string) || "spcm_nn_price_and_sentiment",
      marketDate: p.trading_date,
      observationTimestamp: new Date(p.observation_timestamp),
      isDemoData: false,
    }));

    if (prisma && dbStatus.isConnected) {
      try {
        await prisma.predictionHistory.createMany({
          data: recordsToSave,
        });
      } catch (dbErr) {
        console.warn("[Database] Failed to write predictions to PostgreSQL; fallback to memory store:", dbErr);
        inMemoryStore.recordPredictions(recordsToSave);
      }
    } else {
      inMemoryStore.recordPredictions(recordsToSave);
    }

    return res.json({
      success: true,
      data: predictionResult,
    });
  } catch (err: any) {
    const status = err.response?.status || 503;
    const errorDetail = err.response?.data?.detail || err.message || "Python ML service unavailable";

    return res.status(status).json({
      success: false,
      error: errorDetail,
      service: "python_ml_engine",
      hint: "Ensure the Python FastAPI service is running: py -3.11 -m uvicorn src.inference.fastapi_adapter:app --port 8000",
    });
  }
});

// POST /api/predictions/personalized - Personalized stock recommendations based on Investor Profile
predictionsRouter.post("/personalized", async (req: Request, res: Response) => {
  const parseResult = personalizedRequestSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      success: false,
      error: "Invalid request parameters",
      details: parseResult.error.format(),
    });
  }

  const { profile, symbols, top_k } = parseResult.data;

  // Retrieve base predictions from ML engine or fallback
  let basePredictions: PredictionItem[] = DEMO_FALLBACK_PREDICTIONS;
  let isDemo = true;

  try {
    const pyHealth = await pythonMlService.checkHealth();
    if (pyHealth.online) {
      const predRes = await pythonMlService.runPredict(symbols, top_k);
      basePredictions = predRes.predictions;
      isDemo = false;
    }
  } catch (err) {
    // Fall back to DEMO_FALLBACK_PREDICTIONS
  }

  // Personalization scoring algorithm
  const scored = basePredictions.map((pred) => {
    const stock = pred.stock.toUpperCase();
    const sector = STOCK_SECTORS[stock] || "Other";
    const reasons: string[] = [];

    const normModel = Math.min(1.0, Math.max(0, pred.probability) / 0.35);
    if (pred.probability >= 0.28) reasons.push("High model confidence");

    const isPrefSector = profile.preferredSectors.includes(sector);
    const sectorScore = isPrefSector ? 1.0 : 0.2;
    if (isPrefSector) reasons.push(`Matches your ${sector} focus`);

    let riskScore = 0.5;
    const isDefensive = ["HINDUNILVR", "ITC", "TCS", "HDFCBANK"].includes(stock);
    const isGrowth = ["INFY", "SBIN", "RELIANCE", "BHARTIARTL"].includes(stock);

    if (profile.riskTolerance === "conservative") {
      riskScore = isDefensive ? 1.0 : isGrowth ? 0.4 : 0.7;
      if (isDefensive) reasons.push("Fits Conservative profile (stable large-cap)");
    } else if (profile.riskTolerance === "aggressive") {
      riskScore = isGrowth ? 1.0 : 0.6;
      if (isGrowth) reasons.push("Fits Aggressive profile (high alpha momentum)");
    } else {
      riskScore = 0.85;
      reasons.push("Balanced profile constituent");
    }

    let sentScore = 0.6;
    if (profile.sentimentPreference === "high_sentiment") {
      sentScore = pred.sentiment_label === "bullish" ? 1.0 : pred.sentiment_label === "bearish" ? 0.1 : 0.5;
      if (pred.sentiment_label === "bullish") reasons.push("Bullish news catalyst");
    }

    const totalMatch = Math.round(
      Math.max(10, Math.min(99, (normModel * 0.35 + sectorScore * 0.30 + riskScore * 0.20 + sentScore * 0.15) * 100))
    );

    if (reasons.length === 0) reasons.push("Core NSE constituent");

    return {
      ...pred,
      sector,
      personalizedScore: totalMatch,
      matchReasons: reasons.slice(0, 3),
    };
  });

  scored.sort((a, b) => b.personalizedScore - a.personalizedScore);

  const ranked = scored.map((p, idx) => ({
    ...p,
    personalizedRank: idx + 1,
    isPersonalizedTopPick: idx < top_k,
  }));

  return res.json({
    success: true,
    is_demo_data: isDemo,
    profile,
    top_k,
    top_picks: ranked.slice(0, top_k).map((r) => r.stock),
    recommendations: ranked,
  });
});

// GET /api/predictions/history - Paginated historical records from actual inference runs
predictionsRouter.get("/history", async (req: Request, res: Response) => {
  const stock = req.query.stock as string | undefined;
  const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : 50;
  const offset = req.query.offset ? parseInt(req.query.offset as string, 10) : 0;

  const prisma = getPrismaClient();
  if (prisma && dbStatus.isConnected) {
    try {
      const whereClause = stock ? { stock: stock.toUpperCase() } : {};
      const [total, records] = await Promise.all([
        prisma.predictionHistory.count({ where: whereClause }),
        prisma.predictionHistory.findMany({
          where: whereClause,
          orderBy: { inferenceTimestamp: "desc" },
          take: limit,
          skip: offset,
        }),
      ]);

      return res.json({
        success: true,
        total,
        limit,
        offset,
        source: "postgresql",
        records,
      });
    } catch (err) {
      console.warn("[Database] Error querying PostgreSQL history; falling back to in-memory store:", err);
    }
  }

  // Graceful fallback store
  const { total, records } = inMemoryStore.getPredictionHistory({ stock, limit, offset });
  return res.json({
    success: true,
    total,
    limit,
    offset,
    source: "in_memory_fallback",
    records,
  });
});
