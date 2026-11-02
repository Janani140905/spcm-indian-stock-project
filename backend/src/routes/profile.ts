import { Router, Request, Response } from "express";
import { z } from "zod";
import { getPrismaClient, inMemoryStore, dbStatus } from "../db/prisma.js";
import { DEFAULT_PROFILE } from "../services/recommendationEngine.js";

export const profileRouter = Router();

const profileSchema = z.object({
  riskTolerance: z.enum(["conservative", "moderate", "aggressive"]).default("moderate"),
  preferredSectors: z.array(z.string()).min(1).default(DEFAULT_PROFILE.preferredSectors),
  investmentHorizon: z.enum(["short_term", "medium_term", "long_term"]).default("medium_term"),
  investmentObjective: z
    .enum(["capital_growth", "wealth_preservation", "dividend_income", "balanced"])
    .default("balanced"),
  budgetRange: z
    .enum(["under_25k", "25k_to_100k", "100k_to_500k", "above_500k"])
    .default("25k_to_100k"),
  sentimentPreference: z.enum(["high_sentiment", "balanced", "pure_price"]).default("balanced"),
});

// GET /api/profile - Retrieve investor profile from database or in-memory fallback
profileRouter.get("/", async (_req: Request, res: Response) => {
  const prisma = getPrismaClient();

  if (prisma && dbStatus.isConnected) {
    try {
      const pref = await prisma.appPreference.findUnique({
        where: {
          userId_key: {
            userId: "default_user",
            key: "investor_profile",
          },
        },
      });

      if (pref && pref.value) {
        return res.json({
          success: true,
          source: "postgresql",
          profile: JSON.parse(pref.value),
        });
      }
    } catch (err) {
      console.warn("[Profile] Failed to read from DB, falling back to memory:", err);
    }
  }

  const memProfile = inMemoryStore.getProfile();
  return res.json({
    success: true,
    source: memProfile ? "in_memory_store" : "default_scaffold",
    profile: memProfile || DEFAULT_PROFILE,
  });
});

// POST /api/profile - Save or update investor profile
profileRouter.post("/", async (req: Request, res: Response) => {
  const parseResult = profileSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      success: false,
      error: "Invalid profile data",
      details: parseResult.error.format(),
    });
  }

  const profileData = {
    ...parseResult.data,
    updatedAt: new Date().toISOString(),
  };

  const prisma = getPrismaClient();
  if (prisma && dbStatus.isConnected) {
    try {
      await prisma.appPreference.upsert({
        where: {
          userId_key: {
            userId: "default_user",
            key: "investor_profile",
          },
        },
        update: {
          value: JSON.stringify(profileData),
        },
        create: {
          userId: "default_user",
          key: "investor_profile",
          value: JSON.stringify(profileData),
        },
      });

      return res.json({
        success: true,
        source: "postgresql",
        profile: profileData,
      });
    } catch (err) {
      console.warn("[Profile] Failed to persist to PostgreSQL; saving to in-memory store:", err);
    }
  }

  const saved = inMemoryStore.saveProfile(profileData);
  return res.json({
    success: true,
    source: "in_memory_fallback",
    profile: saved,
  });
});

