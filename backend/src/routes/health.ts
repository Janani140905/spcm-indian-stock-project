import { Router, Request, Response } from "express";
import { pythonMlService } from "../services/pythonMlService.js";
import { dbStatus } from "../db/prisma.js";

export const healthRouter = Router();

healthRouter.get("/", async (_req: Request, res: Response) => {
  const pyHealth = await pythonMlService.checkHealth();

  const response = {
    status: pyHealth.online ? "healthy" : "degraded",
    server: "spcm_express_api",
    timestamp: new Date().toISOString(),
    services: {
      express_api: {
        status: "up",
        port: process.env.PORT || 5000,
        env: process.env.NODE_ENV || "development",
      },
      python_ml_engine: {
        status: pyHealth.online ? "up" : "down",
        url: process.env.PYTHON_ML_SERVICE_URL || "http://127.0.0.1:8000",
        details: pyHealth.data || null,
        error: pyHealth.error || null,
      },
      database: {
        status: dbStatus.isConnected ? "connected" : "in_memory_fallback",
        provider: dbStatus.provider,
        mode: dbStatus.mode,
        error: dbStatus.error || null,
      },
    },
    canonical_universe: [
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
    ],
  };

  return res.status(pyHealth.online ? 200 : 207).json(response);
});
