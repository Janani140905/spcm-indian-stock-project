import { Router, Request, Response } from "express";
import { marketDataService } from "../services/marketDataService.js";
import { pythonMlService } from "../services/pythonMlService.js";

export const stocksRouter = Router();

// GET /api/stocks - List all stocks, with optional sector and text query filters
stocksRouter.get("/", async (req: Request, res: Response) => {
  const sector = (req.query.sector as string) || undefined;
  const query = (req.query.q as string) || (req.query.query as string) || undefined;
  const benchmarkOnly = req.query.benchmark_only === "true";

  const stocks = marketDataService.getStocks(sector, query, benchmarkOnly);

  return res.json({
    count: stocks.length,
    total_universe: marketDataService.getStocks().length,
    sector: sector || "All",
    stocks,
  });
});

// GET /api/stocks/sectors - List all 12 supported sectors with metadata
stocksRouter.get("/sectors", async (_req: Request, res: Response) => {
  const sectors = marketDataService.getSectors();
  return res.json({
    count: sectors.length,
    sectors,
  });
});

// GET /api/stocks/:symbol - Detailed live/delayed quote and technical indicators
stocksRouter.get("/:symbol", async (req: Request, res: Response) => {
  const symbol = req.params.symbol.toUpperCase().replace(".NS", "");

  try {
    const quote = await marketDataService.getStockQuote(symbol);
    return res.json({
      success: true,
      data: quote,
    });
  } catch (err: any) {
    return res.status(404).json({
      success: false,
      error: err.message || `Failed to fetch quote for ${symbol}`,
      symbol,
    });
  }
});

// GET /api/stocks/:symbol/history - Historical OHLCV candle series
stocksRouter.get("/:symbol/history", async (req: Request, res: Response) => {
  const symbol = req.params.symbol.toUpperCase().replace(".NS", "");
  const period = (req.query.period as string) || "6mo";

  try {
    const history = await marketDataService.getStockHistory(symbol, period);
    return res.json({
      success: true,
      data: history,
    });
  } catch (err: any) {
    // If marketDataService threw, attempt pythonMlService fallback
    try {
      const pyHistory = await pythonMlService.getStockHistory(symbol, period);
      return res.json({
        success: true,
        data: pyHistory,
      });
    } catch (pyErr) {
      return res.status(500).json({
        success: false,
        error: err.message || `Failed to fetch history for ${symbol}`,
        symbol,
      });
    }
  }
});
