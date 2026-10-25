import { Router, Request, Response } from "express";
import { z } from "zod";
import { pythonMlService } from "../services/pythonMlService.js";
import { getPrismaClient, inMemoryStore, dbStatus } from "../db/prisma.js";

export const predictionsRouter = Router();

const predictRequestSchema = z.object({
  symbols: z.array(z.string()).optional(),
  top_k: z.number().int().min(1).max(10).default(3),
  warmup_period: z.string().default("60d"),
});

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
