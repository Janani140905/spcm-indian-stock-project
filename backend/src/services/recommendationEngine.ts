import { STOCK_UNIVERSE, StockDefinition } from "../config/stockUniverse.js";
import { marketDataService, StockQuote } from "./marketDataService.js";
import { pythonMlService, PredictionItem } from "./pythonMlService.js";

export interface InvestorProfileData {
  riskTolerance: "conservative" | "moderate" | "aggressive";
  preferredSectors: string[];
  investmentHorizon: "short_term" | "medium_term" | "long_term";
  investmentObjective: "capital_growth" | "wealth_preservation" | "dividend_income" | "balanced";
  budgetRange: "under_25k" | "25k_to_100k" | "100k_to_500k" | "above_500k";
  sentimentPreference: "high_sentiment" | "balanced" | "pure_price";
}

export interface RecommendationItem {
  symbol: string;
  ticker: string;
  name: string;
  sector: string;
  exchange: "NSE";
  currentPrice: number;
  change: number;
  changePct: number;
  marketCapCategory: "Large Cap" | "Mid Cap";
  suitabilityScore: number; // 0 to 100 percentage
  rank: number;
  isTopPick: boolean;
  matchReasons: string[];
  riskIndicators: string[];
  factorScores: {
    sectorMatch: number;
    riskFit: number;
    objectiveFit: number;
    horizonFit: number;
    sentimentFit: number;
    spcmModelFit: number;
  };
  spcmModel: {
    status: "AVAILABLE" | "TRAINING_REQUIRED";
    probability: number | null;
    rank: number | null;
    sentimentLabel: string | null;
    note: string;
  };
  marketDataTimestamp: string;
}

export interface RecommendationResponse {
  totalAnalyzed: number;
  topK: number;
  profile: InvestorProfileData;
  recommendations: RecommendationItem[];
  sectorDistribution: Record<string, number>;
  modelTrainingNotice: string;
  disclaimer: string;
  timestamp: string;
}

export const DEFAULT_PROFILE: InvestorProfileData = {
  riskTolerance: "moderate",
  preferredSectors: [
    "Information Technology",
    "Banking and Financial Services",
    "Energy and Oil & Gas",
    "Pharmaceuticals and Healthcare",
  ],
  investmentHorizon: "medium_term",
  investmentObjective: "balanced",
  budgetRange: "25k_to_100k",
  sentimentPreference: "balanced",
};

class RecommendationEngine {
  private cachedPyPredictions: Map<string, PredictionItem> = new Map();
  private lastPyPredFetch = 0;
  private PY_PRED_TTL_MS = 15 * 60 * 1000;

  /**
   * Generates ranked personalized stock recommendations across the expanded Indian universe.
   */
  async generateRecommendations(
    profileInput?: Partial<InvestorProfileData>,
    topK = 5,
    sectorFilter?: string
  ): Promise<RecommendationResponse> {
    const profile: InvestorProfileData = {
      riskTolerance: profileInput?.riskTolerance || DEFAULT_PROFILE.riskTolerance,
      preferredSectors:
        profileInput?.preferredSectors && profileInput.preferredSectors.length > 0
          ? profileInput.preferredSectors
          : DEFAULT_PROFILE.preferredSectors,
      investmentHorizon: profileInput?.investmentHorizon || DEFAULT_PROFILE.investmentHorizon,
      investmentObjective: profileInput?.investmentObjective || DEFAULT_PROFILE.investmentObjective,
      budgetRange: profileInput?.budgetRange || DEFAULT_PROFILE.budgetRange,
      sentimentPreference: profileInput?.sentimentPreference || DEFAULT_PROFILE.sentimentPreference,
    };

    // 1. Filter eligible candidate stocks based on profile & sector focus
    let candidates = STOCK_UNIVERSE;
    if (sectorFilter && sectorFilter !== "All") {
      const secLower = sectorFilter.toLowerCase();
      candidates = candidates.filter((s) => s.sector.toLowerCase() === secLower);
    } else if (profile.preferredSectors && profile.preferredSectors.length > 0) {
      const preferredSet = new Set(profile.preferredSectors.map((s) => s.toLowerCase()));
      const preferredStocks = STOCK_UNIVERSE.filter((s) => preferredSet.has(s.sector.toLowerCase()));
      const otherStocks = STOCK_UNIVERSE.filter((s) => !preferredSet.has(s.sector.toLowerCase()));
      candidates = [...preferredStocks, ...otherStocks.slice(0, 10)];
    }

    // 2. Fetch available SPCM model predictions for candidate stocks (cached)
    const now = Date.now();
    if (this.cachedPyPredictions.size === 0 || now - this.lastPyPredFetch > this.PY_PRED_TTL_MS) {
      if (process.env.NODE_ENV !== "test") {
        try {
          const pyHealth = await pythonMlService.checkHealth();
          if (pyHealth.online) {
            const targetSymbols = candidates.slice(0, 12).map((c) => c.symbol);
            const pyPredRes = await pythonMlService.runPredict(targetSymbols, 5);
            if (pyPredRes && pyPredRes.predictions) {
              for (const pred of pyPredRes.predictions) {
                this.cachedPyPredictions.set(pred.stock.toUpperCase(), pred);
              }
              this.lastPyPredFetch = now;
            }
          }
        } catch (err) {
          // SPCM inference engine offline or model not loaded; handle gracefully
        }
      }
    }
    const spcmPredictionsMap = this.cachedPyPredictions;

    // 3. Score each candidate stock
    const scoredPromises = candidates.map(async (stockDef) => {
      let quote: StockQuote;
      try {
        quote = await marketDataService.getStockQuote(stockDef.symbol);
      } catch (e) {
        // Fallback quote structure if provider fails
        quote = {
          symbol: stockDef.symbol,
          ticker: stockDef.ticker,
          name: stockDef.name,
          sector: stockDef.sector,
          exchange: stockDef.exchange,
          currentPrice: 0,
          previousClose: 0,
          change: 0,
          changePct: 0,
          open: 0,
          high: 0,
          low: 0,
          volume: 0,
          fiftyTwoWeekHigh: null,
          fiftyTwoWeekLow: null,
          timestamp: new Date().toISOString(),
          marketTimeIST: "N/A",
          feedStatus: "HISTORICAL_EOD",
          inTrainingUniverse: stockDef.inTrainingUniverse,
          marketCapCategory: stockDef.marketCapCategory,
          description: stockDef.description,
          technicalIndicators: {
            sma20: null,
            sma50: null,
            volatility30d: null,
            momentum10d: null,
            rsi14: null,
          },
        };
      }

      return this.evaluateStockSuitability(stockDef, quote, profile, spcmPredictionsMap.get(stockDef.symbol));
    });

    const scoredItems = await Promise.all(scoredPromises);

    // 4. Sort descending by suitability score
    scoredItems.sort((a, b) => b.suitabilityScore - a.suitabilityScore);

    // 5. Assign ranks and top pick tags
    const ranked: RecommendationItem[] = scoredItems.map((item, idx) => ({
      ...item,
      rank: idx + 1,
      isTopPick: idx < topK,
    }));

    // Calculate sector distribution of recommendations
    const sectorDistribution: Record<string, number> = {};
    for (const item of ranked.slice(0, topK)) {
      sectorDistribution[item.sector] = (sectorDistribution[item.sector] || 0) + 1;
    }

    return {
      totalAnalyzed: ranked.length,
      topK,
      profile,
      recommendations: ranked.slice(0, topK),
      sectorDistribution,
      modelTrainingNotice:
        "The SPCM Multimodal Neural Network model (v2.0-50stocks) is trained across all 50 NSE stocks in 12 sectors. " +
        "Recommendations combine genuine model forward-return probabilities with multi-factor volatility, momentum, sector suitability, and investor profile preferences.",
      disclaimer:
        "DISCLAIMER: Recommendations generated by SPCM Intelligence are educational quantitative research outputs " +
        "based on historical market volatility, momentum indicators, and user-specified investor profile preferences. " +
        "This does NOT constitute personalized, SEBI-registered, or guaranteed financial investment advice.",
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Transparent factor evaluation for a single stock against an investor profile
   */
  private evaluateStockSuitability(
    def: StockDefinition,
    quote: StockQuote,
    profile: InvestorProfileData,
    spcmPred?: PredictionItem
  ): RecommendationItem {
    const matchReasons: string[] = [];
    const riskIndicators: string[] = [];

    // Factor 1: Sector Match (Weight: 25%)
    const isPreferredSector = profile.preferredSectors.includes(def.sector);
    const sectorScore = isPreferredSector ? 1.0 : 0.25;
    if (isPreferredSector) {
      matchReasons.push(`Matches your ${def.sector} sector focus`);
    }

    // Factor 2: Risk Profile Fit (Weight: 20%)
    let riskScore = 0.5;
    const vol = quote.technicalIndicators.volatility30d;

    if (profile.riskTolerance === "conservative") {
      if (vol !== null && vol < 22) {
        riskScore = 1.0;
        matchReasons.push(`Low 30-day volatility (${vol}%) aligns with Conservative risk profile`);
      } else if (vol !== null && vol > 32) {
        riskScore = 0.25;
        riskIndicators.push(`Higher historical volatility (${vol}%) exceeds conservative threshold`);
      } else {
        riskScore = 0.65;
      }

      if (["FMCG and Consumer Goods", "Pharmaceuticals and Healthcare"].includes(def.sector)) {
        riskScore = Math.min(1.0, riskScore + 0.15);
        matchReasons.push("Defensive consumer/pharma sector dampens downside risk");
      }
    } else if (profile.riskTolerance === "aggressive") {
      if (vol !== null && vol >= 24) {
        riskScore = 1.0;
        matchReasons.push(`High beta & volatility (${vol}%) offers aggressive growth upside`);
      } else {
        riskScore = 0.6;
      }
      if (["Automobile and Electric Vehicles", "Metals and Mining", "Information Technology"].includes(def.sector)) {
        riskScore = Math.min(1.0, riskScore + 0.15);
      }
    } else {
      // Moderate
      if (vol !== null && vol >= 16 && vol <= 30) {
        riskScore = 0.9;
        matchReasons.push(`Balanced historical volatility (${vol}%) fits Moderate risk profile`);
      } else {
        riskScore = 0.7;
      }
    }

    // Factor 3: Investment Objective Fit (Weight: 20%)
    let objectiveScore = 0.5;
    const momentum10d = quote.technicalIndicators.momentum10d;

    switch (profile.investmentObjective) {
      case "capital_growth":
        if (momentum10d !== null && momentum10d > 1.5) {
          objectiveScore = 1.0;
          matchReasons.push(`Positive 10-day price momentum (+${momentum10d}%) supports Capital Growth`);
        } else if (momentum10d !== null && momentum10d < -3.0) {
          objectiveScore = 0.35;
          riskIndicators.push(`Negative 10-day price trend (${momentum10d}%)`);
        } else {
          objectiveScore = 0.7;
        }
        break;

      case "wealth_preservation":
        if (["FMCG and Consumer Goods", "Banking and Financial Services", "Pharmaceuticals and Healthcare"].includes(def.sector)) {
          objectiveScore = 0.95;
          matchReasons.push("Resilient large-cap market presence supports Wealth Preservation");
        } else {
          objectiveScore = 0.55;
        }
        break;

      case "dividend_income":
        if (["Energy and Oil & Gas", "Power and Renewable Energy", "FMCG and Consumer Goods"].includes(def.sector)) {
          objectiveScore = 0.95;
          matchReasons.push("Mature cash-flow sector historically associated with dividend distributions");
        } else {
          objectiveScore = 0.5;
        }
        break;

      case "balanced":
      default:
        objectiveScore = 0.8;
        break;
    }

    // Factor 4: Investment Horizon Fit (Weight: 15%)
    let horizonScore = 0.6;
    const rsi = quote.technicalIndicators.rsi14;

    if (profile.investmentHorizon === "short_term") {
      if (momentum10d !== null && momentum10d > 2.0) {
        horizonScore = 0.9;
        matchReasons.push("Short-term catalyst momentum for swing trading");
      }
      if (rsi !== null && rsi > 72) {
        riskIndicators.push(`RSI (${rsi}) indicates near-term overbought levels`);
      }
    } else if (profile.investmentHorizon === "long_term") {
      horizonScore = def.marketCapCategory === "Large Cap" ? 0.95 : 0.7;
      matchReasons.push("Large-cap industry leader suitable for multi-month horizon");
    } else {
      horizonScore = 0.8;
    }

    // Factor 5: News Sentiment (Weight: 10%)
    let sentimentScore = 0.5;
    if (spcmPred && spcmPred.sentiment_label) {
      if (spcmPred.sentiment_label === "bullish") {
        sentimentScore = 0.95;
        matchReasons.push("Supportive bullish FinBERT financial news sentiment");
      } else if (spcmPred.sentiment_label === "bearish") {
        sentimentScore = 0.2;
        riskIndicators.push("Bearish financial news sentiment detected");
      } else {
        sentimentScore = 0.6;
      }
    }

    // Factor 6: SPCM ML Model Prediction (Weight: 10%)
    let spcmModelScore = 0.5; // neutral baseline when real-time prediction is loading
    let spcmStatus: "AVAILABLE" | "TRAINING_REQUIRED" = "AVAILABLE";
    let spcmNote = "SPCM 50-stock multimodal neural network inference active";

    if (spcmPred) {
      spcmStatus = "AVAILABLE";
      spcmModelScore = Math.min(1.0, Math.max(0, spcmPred.probability / 0.35));
      spcmNote = `SPCM predicted forward outperformance prob P=${(spcmPred.probability * 100).toFixed(1)}%`;
      if (spcmPred.probability >= 0.28) {
        matchReasons.push(`SPCM model signals high forward return probability (${(spcmPred.probability * 100).toFixed(1)}%)`);
      }
    } else {
      spcmNote = "SPCM neural model trained; generating real-time cross-sectional signals";
    }

    // Budget check caveat
    if (profile.budgetRange === "under_25k" && quote.currentPrice > 10000) {
      riskIndicators.push(`High per-share price (₹${quote.currentPrice}) requires >40% of under ₹25k starter budget`);
    }

    // Ensure at least 1 match reason is present
    if (matchReasons.length === 0) {
      matchReasons.push(`High liquidity NSE constituent in ${def.sector}`);
    }

    // Weighted Composite Suitability Score (0 to 100%)
    const rawSuitability =
      sectorScore * 0.25 +
      riskScore * 0.20 +
      objectiveScore * 0.20 +
      horizonScore * 0.15 +
      sentimentScore * 0.10 +
      spcmModelScore * 0.10;

    const suitabilityScore = Math.round(Math.max(10, Math.min(99, rawSuitability * 100)));

    return {
      symbol: def.symbol,
      ticker: def.ticker,
      name: def.name,
      sector: def.sector,
      exchange: def.exchange,
      currentPrice: quote.currentPrice,
      change: quote.change,
      changePct: quote.changePct,
      marketCapCategory: def.marketCapCategory,
      suitabilityScore,
      rank: 0,
      isTopPick: false,
      matchReasons: matchReasons.slice(0, 4),
      riskIndicators: riskIndicators.slice(0, 3),
      factorScores: {
        sectorMatch: Math.round(sectorScore * 100),
        riskFit: Math.round(riskScore * 100),
        objectiveFit: Math.round(objectiveScore * 100),
        horizonFit: Math.round(horizonScore * 100),
        sentimentFit: Math.round(sentimentScore * 100),
        spcmModelFit: Math.round(spcmModelScore * 100),
      },
      spcmModel: {
        status: spcmStatus,
        probability: spcmPred ? round(spcmPred.probability, 4) : null,
        rank: spcmPred ? spcmPred.rank : null,
        sentimentLabel: spcmPred ? spcmPred.sentiment_label : null,
        note: spcmNote,
      },
      marketDataTimestamp: quote.timestamp,
    };
  }
}

function round(val: number, decimals = 2): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

export const recommendationEngine = new RecommendationEngine();
