import { Router, Request, Response } from "express";
import { pythonMlService } from "../services/pythonMlService.js";

export const stocksRouter = Router();

// Fallback stock catalog when python service is offline
const FALLBACK_STOCKS = [
  { symbol: "RELIANCE", ticker: "RELIANCE.NS", name: "Reliance Industries Ltd", sector: "Energy & Conglomerate", exchange: "NSE" },
  { symbol: "TCS", ticker: "TCS.NS", name: "Tata Consultancy Services Ltd", sector: "Information Technology", exchange: "NSE" },
  { symbol: "HDFCBANK", ticker: "HDFCBANK.NS", name: "HDFC Bank Ltd", sector: "Banking & Financial Services", exchange: "NSE" },
  { symbol: "INFY", ticker: "INFY.NS", name: "Infosys Ltd", sector: "Information Technology", exchange: "NSE" },
  { symbol: "ICICIBANK", ticker: "ICICIBANK.NS", name: "ICICI Bank Ltd", sector: "Banking & Financial Services", exchange: "NSE" },
  { symbol: "HINDUNILVR", ticker: "HINDUNILVR.NS", name: "Hindustan Unilever Ltd", sector: "Consumer Goods (FMCG)", exchange: "NSE" },
  { symbol: "ITC", ticker: "ITC.NS", name: "ITC Ltd", sector: "Consumer Goods & Conglomerate", exchange: "NSE" },
  { symbol: "SBIN", ticker: "SBIN.NS", name: "State Bank of India", sector: "Public Sector Banking", exchange: "NSE" },
  { symbol: "BHARTIARTL", ticker: "BHARTIARTL.NS", name: "Bharti Airtel Ltd", sector: "Telecommunications", exchange: "NSE" },
  { symbol: "KOTAKBANK", ticker: "KOTAKBANK.NS", name: "Kotak Mahindra Bank Ltd", sector: "Banking & Financial Services", exchange: "NSE" },
];

stocksRouter.get("/", async (_req: Request, res: Response) => {
  try {
    const pyHealth = await pythonMlService.checkHealth();
    if (pyHealth.online) {
      const stocks = await pythonMlService.getStocks();
      return res.json({
        count: stocks.length,
        source: "python_ml_service",
        stocks,
      });
    }
  } catch (err) {
    // Graceful fallback to static catalog
  }

  return res.json({
    count: FALLBACK_STOCKS.length,
    source: "static_catalog",
    stocks: FALLBACK_STOCKS,
  });
});

stocksRouter.get("/:symbol/history", async (req: Request, res: Response) => {
  const symbol = req.params.symbol.toUpperCase();
  const period = (req.query.period as string) || "6mo";

  try {
    const history = await pythonMlService.getStockHistory(symbol, period);
    return res.json({
      success: true,
      data: history,
    });
  } catch (err: any) {
    const isConnRefused = err.code === "ECONNREFUSED" || err.message?.includes("ECONNREFUSED");
    const status = err.response?.status || (isConnRefused ? 503 : 500);
    const msg = err.response?.data?.detail || err.message || "Failed to fetch stock history";
    return res.status(status).json({
      success: false,
      error: msg,
      symbol,
      service: isConnRefused ? "python_ml_engine" : undefined,
      hint: isConnRefused ? "Python ML service offline on port 8000" : undefined,
    });
  }
});
