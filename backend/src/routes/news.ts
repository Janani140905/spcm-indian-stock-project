import { Router, Request, Response } from "express";
import { pythonMlService } from "../services/pythonMlService.js";

export const newsRouter = Router();

newsRouter.get("/", async (req: Request, res: Response) => {
  const symbolsParam = req.query.symbols as string | undefined;
  const limitParam = req.query.limit ? parseInt(req.query.limit as string, 10) : 30;

  const symbols = symbolsParam
    ? symbolsParam.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean)
    : undefined;

  try {
    const newsData = await pythonMlService.getNews(symbols, limitParam);
    return res.json({
      success: true,
      data: newsData,
    });
  } catch (err: any) {
    const isConnRefused = err.code === "ECONNREFUSED" || err.message?.includes("ECONNREFUSED");
    const status = err.response?.status || (isConnRefused ? 503 : 500);
    const msg = err.response?.data?.detail || err.message || "Failed to fetch news";
    return res.status(status).json({
      success: false,
      error: msg,
      service: isConnRefused ? "python_ml_engine" : undefined,
      hint: isConnRefused ? "Python ML service offline on port 8000" : undefined,
    });
  }
});
