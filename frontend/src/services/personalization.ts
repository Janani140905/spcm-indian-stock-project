import {
  Prediction,
  InvestorProfile,
  PersonalizedPrediction,
  RiskTolerance,
  InvestmentHorizon,
  InvestmentObjective,
  BudgetRange,
  SentimentPreference,
} from "../types/index.js";
import { SECTORS, STOCK_UNIVERSE } from "../config/stockUniverse.js";

export const AVAILABLE_SECTORS = SECTORS.map((s) => ({
  id: s.name,
  label: s.displayName,
  shortCode: s.shortCode,
  description: s.description,
  stocks: s.stockSymbols,
}));

export const DEFAULT_INVESTOR_PROFILE: InvestorProfile = {
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
  updatedAt: new Date().toISOString(),
};

export const STORAGE_KEY = "spcm_investor_profile";

export function loadStoredProfile(): InvestorProfile | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return {
      ...DEFAULT_INVESTOR_PROFILE,
      ...parsed,
    };
  } catch (e) {
    console.warn("Failed to load investor profile from localStorage:", e);
    return null;
  }
}

export function saveStoredProfile(profile: InvestorProfile): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(profile));
  } catch (e) {
    console.warn("Failed to save investor profile to localStorage:", e);
  }
}

export function clearStoredProfile(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (e) {
    console.warn("Failed to clear investor profile:", e);
  }
}

export const STOCK_SECTOR_MAP: Record<string, string> = STOCK_UNIVERSE.reduce(
  (acc, s) => {
    acc[s.symbol.toUpperCase()] = s.sector;
    return acc;
  },
  {} as Record<string, string>
);

/**
 * Re-ranks SPCM predictions based on user tastes, risk tolerance, and sector interest.
 */
export function computePersonalizedRankings(
  predictions: Prediction[],
  profile: InvestorProfile
): PersonalizedPrediction[] {
  if (!predictions || predictions.length === 0) return [];

  const scored: PersonalizedPrediction[] = predictions.map((pred) => {
    const stock = pred.stock.toUpperCase();
    const sector = STOCK_SECTOR_MAP[stock] || "Other";
    const reasons: string[] = [];

    // 1. SPCM Base Model Probability (Weight: 35%)
    const baseModelProb = Math.max(0, Math.min(1, pred.probability));
    const normalizedModelScore = Math.min(1.0, baseModelProb / 0.35);

    if (pred.probability >= 0.28) {
      reasons.push("High SPCM model confidence");
    }

    // 2. Sector Match Score (Weight: 30%)
    const isPreferredSector = profile.preferredSectors.includes(sector);
    let sectorScore = isPreferredSector ? 1.0 : 0.25;
    if (isPreferredSector) {
      reasons.push(`Matches your ${sector} focus`);
    }

    // 3. Risk Profile Fit (Weight: 20%)
    let riskScore = 0.5;
    const isDefensive = [
      "HINDUNILVR",
      "ITC",
      "NESTLEIND",
      "BRITANNIA",
      "SUNPHARMA",
      "CIPLA",
      "TCS",
      "HDFCBANK",
    ].includes(stock);
    const isGrowthMomentum = [
      "INFY",
      "SBIN",
      "RELIANCE",
      "TATAMOTORS",
      "TRENT",
      "BHARTIARTL",
    ].includes(stock);

    if (profile.riskTolerance === "conservative") {
      if (isDefensive) {
        riskScore = 1.0;
        reasons.push("Fits Conservative profile (stable large-cap)");
      } else if (isGrowthMomentum) {
        riskScore = 0.4;
      } else {
        riskScore = 0.7;
      }
    } else if (profile.riskTolerance === "aggressive") {
      if (isGrowthMomentum || pred.probability > 0.26) {
        riskScore = 1.0;
        reasons.push("Fits Aggressive profile (high alpha momentum)");
      } else {
        riskScore = 0.6;
      }
    } else {
      riskScore = 0.85;
      reasons.push("Balanced profile constituent");
    }

    // 4. Sentiment & Investment Horizon Fit (Weight: 15%)
    let sentimentHorizonScore = 0.5;
    const isBullishSentiment =
      pred.sentiment_label === "bullish" ||
      (pred.sent_score !== null && pred.sent_score > 0.2);
    const isBearishSentiment =
      pred.sentiment_label === "bearish" ||
      (pred.sent_score !== null && pred.sent_score < -0.2);

    if (profile.sentimentPreference === "high_sentiment") {
      if (isBullishSentiment) {
        sentimentHorizonScore = 1.0;
        reasons.push("Strong positive FinBERT news sentiment");
      } else if (isBearishSentiment) {
        sentimentHorizonScore = 0.1;
      } else {
        sentimentHorizonScore = 0.5;
      }
    } else if (profile.sentimentPreference === "pure_price") {
      sentimentHorizonScore = normalizedModelScore;
    } else {
      if (isBullishSentiment) {
        sentimentHorizonScore = 0.9;
        reasons.push("Supportive news sentiment");
      } else if (isBearishSentiment) {
        sentimentHorizonScore = 0.3;
      } else {
        sentimentHorizonScore = 0.7;
      }
    }

    if (profile.investmentHorizon === "short_term" && pred.news_count > 0 && isBullishSentiment) {
      reasons.push("High news catalyst activity for swing trading");
    }

    const rawMatch =
      normalizedModelScore * 0.35 +
      sectorScore * 0.30 +
      riskScore * 0.20 +
      sentimentHorizonScore * 0.15;

    const personalizedScore = Math.round(Math.max(10, Math.min(99, rawMatch * 100)));

    if (reasons.length === 0) {
      reasons.push("Core NSE constituent");
    }

    return {
      ...pred,
      personalizedScore,
      personalizedRank: 0,
      matchReasons: reasons.slice(0, 3),
      sector,
    };
  });

  scored.sort((a, b) => b.personalizedScore - a.personalizedScore);

  return scored.map((item, idx) => ({
    ...item,
    personalizedRank: idx + 1,
  }));
}
