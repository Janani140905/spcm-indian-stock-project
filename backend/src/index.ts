import express, { Request, Response, NextFunction } from "express";
import cors from "cors";
import { env } from "./config/env.js";
import { initDatabase } from "./db/prisma.js";
import { healthRouter } from "./routes/health.js";
import { stocksRouter } from "./routes/stocks.js";
import { predictionsRouter } from "./routes/predictions.js";
import { newsRouter } from "./routes/news.js";
import { watchlistRouter } from "./routes/watchlist.js";

const app = express();

// CORS configuration
const allowedOrigins = env.CORS_ORIGIN.split(",").map((s) => s.trim());
app.use(
  cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (e.g. mobile apps, curl, server-to-server)
      if (!origin || allowedOrigins.includes(origin) || origin.includes("localhost") || origin.includes("127.0.0.1")) {
        return callback(null, true);
      }
      return callback(new Error(`CORS blocked origin: ${origin}`));
    },
    credentials: true,
  })
);

app.use(express.json());

// Request logger
app.use((req: Request, _res: Response, next: NextFunction) => {
  const start = Date.now();
  _res.on("finish", () => {
    const elapsed = Date.now() - start;
    console.log(`[HTTP] ${req.method} ${req.originalUrl} ${_res.statusCode} (${elapsed}ms)`);
  });
  next();
});

// Mount API routes
app.use("/api/health", healthRouter);
app.use("/api/stocks", stocksRouter);
app.use("/api/predictions", predictionsRouter);
app.use("/api/news", newsRouter);
app.use("/api/watchlist", watchlistRouter);

// Root greeting
app.get("/", (_req: Request, res: Response) => {
  res.json({
    message: "SPCM Intelligence - Indian Stock Prediction API (Express + Prisma)",
    status: "online",
    docs: "/api/health",
  });
});

// Centralized error handler
app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  console.error("[Error] Unhandled API error:", err);
  const status = err.status || 500;
  res.status(status).json({
    success: false,
    error: err.message || "Internal server error",
  });
});

// Server bootstrap
const port = env.PORT;
if (process.env.NODE_ENV !== "test") {
  app.listen(port, async () => {
    console.log(`=======================================================`);
    console.log(` SPCM Intelligence Backend (Express + TypeScript)`);
    console.log(` Server running on http://localhost:${port}`);
    console.log(` Python ML Service: ${env.PYTHON_ML_SERVICE_URL}`);
    console.log(`=======================================================`);
    await initDatabase();
  });
}

export default app;
