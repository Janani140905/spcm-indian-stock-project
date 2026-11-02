import { Router, Request, Response } from "express";
import { z } from "zod";
import { recommendationEngine, DEFAULT_PROFILE } from "../services/recommendationEngine.js";
import { inMemoryStore, getPrismaClient, dbStatus } from "../db/prisma.js";

export const recommendationsRouter = Router();

const recommendRequestSchema = z.object({
  profile: z
    .object({
      riskTolerance: z.enum(["conservative", "moderate", "aggressive"]).optional(),
      preferredSectors: z.array(z.string()).optional(),
      investmentHorizon: z.enum(["short_term", "medium_term", "long_term"]).optional(),
      investmentObjective: z
        .enum(["capital_growth", "wealth_preservation", "dividend_income", "balanced"])
        .optional(),
      budgetRange: z
        .enum(["under_25k", "25k_to_100k", "100k_to_500k", "above_500k"])
        .optional(),
      sentimentPreference: z.enum(["high_sentiment", "balanced", "pure_price"]).optional(),
    })
    .optional(),
  top_k: z.number().int().min(1).max(20).default(5),
  sector: z.string().optional(),
});

// GET /api/recommendations - Generate recommendations using active profile
recommendationsRouter.get("/", async (req: Request, res: Response) => {
  const topK = req.query.top_k ? parseInt(req.query.top_k as string, 10) : 5;
  const sector = (req.query.sector as string) || undefined;

  // Attempt to load saved profile
  let activeProfile = inMemoryStore.getProfile();
  const prisma = getPrismaClient();
  if (!activeProfile && prisma && dbStatus.isConnected) {
    try {
      const pref = await prisma.appPreference.findUnique({
        where: { userId_key: { userId: "default_user", key: "investor_profile" } },
      });
      if (pref && pref.value) {
        activeProfile = JSON.parse(pref.value);
      }
    } catch (e) {
      // Ignore
    }
  }

  try {
    const result = await recommendationEngine.generateRecommendations(activeProfile || DEFAULT_PROFILE, topK, sector);
    return res.json({
      success: true,
      data: result,
    });
  } catch (err: any) {
    return res.status(500).json({
      success: false,
      error: err.message || "Failed to generate recommendations",
    });
  }
});

// POST /api/recommendations - Generate recommendations for specified profile
recommendationsRouter.post("/", async (req: Request, res: Response) => {
  const parseResult = recommendRequestSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      success: false,
      error: "Invalid recommendation request parameters",
      details: parseResult.error.format(),
    });
  }

  const { profile, top_k, sector } = parseResult.data;

  try {
    const result = await recommendationEngine.generateRecommendations(profile, top_k, sector);
    return res.json({
      success: true,
      data: result,
    });
  } catch (err: any) {
    return res.status(500).json({
      success: false,
      error: err.message || "Failed to generate personalized recommendations",
    });
  }
});

