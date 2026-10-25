-- CreateTable
CREATE TABLE "users" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "email" TEXT NOT NULL,
    "name" TEXT,
    "passwordHash" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL
);

-- CreateTable
CREATE TABLE "watchlists" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "name" TEXT NOT NULL DEFAULT 'Default Watchlist',
    "userId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "watchlists_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "watchlist_items" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "watchlistId" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "ticker" TEXT NOT NULL,
    "targetPrice" DOUBLE PRECISION,
    "notes" TEXT,
    "addedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "watchlist_items_watchlistId_fkey" FOREIGN KEY ("watchlistId") REFERENCES "watchlists" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "prediction_history" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "stock" TEXT NOT NULL,
    "symbol" TEXT NOT NULL,
    "closePrice" DOUBLE PRECISION NOT NULL,
    "probability" DOUBLE PRECISION NOT NULL,
    "rank" INTEGER NOT NULL,
    "isRecommended" BOOLEAN NOT NULL,
    "sentimentLabel" TEXT NOT NULL,
    "sentScore" DOUBLE PRECISION,
    "pvVolume" DOUBLE PRECISION,
    "aiIndex" DOUBLE PRECISION,
    "newsCount" INTEGER NOT NULL DEFAULT 0,
    "modelVersion" TEXT NOT NULL DEFAULT 'spcm_nn_price_and_sentiment',
    "marketDate" TEXT NOT NULL,
    "observationTimestamp" TIMESTAMP(3) NOT NULL,
    "inferenceTimestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "isDemoData" BOOLEAN NOT NULL DEFAULT false
);

-- CreateTable
CREATE TABLE "app_preferences" (
    "id" TEXT NOT NULL PRIMARY KEY,
    "userId" TEXT,
    "key" TEXT NOT NULL,
    "value" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    CONSTRAINT "app_preferences_userId_fkey" FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "users"("email");

-- CreateIndex
CREATE INDEX "watchlist_items_symbol_idx" ON "watchlist_items"("symbol");

-- CreateIndex
CREATE UNIQUE INDEX "watchlist_items_watchlistId_symbol_key" ON "watchlist_items"("watchlistId", "symbol");

-- CreateIndex
CREATE INDEX "prediction_history_stock_idx" ON "prediction_history"("stock");

-- CreateIndex
CREATE INDEX "prediction_history_inferenceTimestamp_idx" ON "prediction_history"("inferenceTimestamp");

-- CreateIndex
CREATE UNIQUE INDEX "app_preferences_userId_key_key" ON "app_preferences"("userId", "key");
