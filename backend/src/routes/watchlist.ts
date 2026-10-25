import { Router, Request, Response } from "express";
import { z } from "zod";
import { getPrismaClient, inMemoryStore, dbStatus } from "../db/prisma.js";

export const watchlistRouter = Router();

const addWatchlistSchema = z.object({
  symbol: z.string().min(1),
  ticker: z.string().optional(),
  targetPrice: z.number().optional().nullable(),
  notes: z.string().optional().nullable(),
});

// GET /api/watchlist
watchlistRouter.get("/", async (_req: Request, res: Response) => {
  const prisma = getPrismaClient();

  if (prisma && dbStatus.isConnected) {
    try {
      // Find or create default watchlist
      let defaultWatchlist = await prisma.watchlist.findFirst({
        include: { items: true },
      });
      if (!defaultWatchlist) {
        defaultWatchlist = await prisma.watchlist.create({
          data: { name: "Default Watchlist" },
          include: { items: true },
        });
      }
      return res.json({
        success: true,
        source: "postgresql",
        items: defaultWatchlist.items,
      });
    } catch (err) {
      console.warn("[Database] Failed to read watchlist from PostgreSQL; fallback to memory store:", err);
    }
  }

  return res.json({
    success: true,
    source: "in_memory_fallback",
    items: inMemoryStore.getWatchlist(),
  });
});

// POST /api/watchlist
watchlistRouter.post("/", async (req: Request, res: Response) => {
  const parseResult = addWatchlistSchema.safeParse(req.body);
  if (!parseResult.success) {
    return res.status(400).json({
      success: false,
      error: "Invalid watchlist payload",
      details: parseResult.error.format(),
    });
  }

  const { symbol, targetPrice, notes } = parseResult.data;
  const cleanSymbol = symbol.toUpperCase().replace(".NS", "");
  const cleanTicker = `${cleanSymbol}.NS`;

  const prisma = getPrismaClient();
  if (prisma && dbStatus.isConnected) {
    try {
      let defaultWatchlist = await prisma.watchlist.findFirst();
      if (!defaultWatchlist) {
        defaultWatchlist = await prisma.watchlist.create({
          data: { name: "Default Watchlist" },
        });
      }

      const item = await prisma.watchlistItem.upsert({
        where: {
          watchlistId_symbol: {
            watchlistId: defaultWatchlist.id,
            symbol: cleanSymbol,
          },
        },
        update: {
          targetPrice: targetPrice ?? undefined,
          notes: notes ?? undefined,
        },
        create: {
          watchlistId: defaultWatchlist.id,
          symbol: cleanSymbol,
          ticker: cleanTicker,
          targetPrice: targetPrice ?? null,
          notes: notes ?? null,
        },
      });

      return res.status(201).json({
        success: true,
        source: "postgresql",
        item,
      });
    } catch (err) {
      console.warn("[Database] Failed to save watchlist to PostgreSQL; fallback to memory store:", err);
    }
  }

  const item = inMemoryStore.addWatchlistItem({
    symbol: cleanSymbol,
    ticker: cleanTicker,
    targetPrice,
    notes,
  });

  return res.status(201).json({
    success: true,
    source: "in_memory_fallback",
    item,
  });
});

// DELETE /api/watchlist/:symbol
watchlistRouter.delete("/:symbol", async (req: Request, res: Response) => {
  const cleanSymbol = req.params.symbol.toUpperCase().replace(".NS", "");
  const prisma = getPrismaClient();

  if (prisma && dbStatus.isConnected) {
    try {
      const defaultWatchlist = await prisma.watchlist.findFirst();
      if (defaultWatchlist) {
        await prisma.watchlistItem.deleteMany({
          where: {
            watchlistId: defaultWatchlist.id,
            symbol: cleanSymbol,
          },
        });
      }
      return res.json({
        success: true,
        message: `Removed ${cleanSymbol} from watchlist`,
        symbol: cleanSymbol,
      });
    } catch (err) {
      console.warn("[Database] Failed to delete from PostgreSQL; fallback to memory store:", err);
    }
  }

  inMemoryStore.removeWatchlistItem(cleanSymbol);
  return res.json({
    success: true,
    message: `Removed ${cleanSymbol} from watchlist`,
    symbol: cleanSymbol,
  });
});
