import request from "supertest";
import app from "../index.js";

describe("SPCM Intelligence Express API Route Tests", () => {
  it("GET /api/health should return health payload", async () => {
    const res = await request(app).get("/api/health");
    expect([200, 207]).toContain(res.status);
    expect(res.body).toHaveProperty("server", "spcm_express_api");
    expect(res.body.services).toHaveProperty("express_api");
    expect(res.body.services).toHaveProperty("python_ml_engine");
    expect(res.body.services).toHaveProperty("database");
    expect(res.body.canonical_universe.length).toBe(10);
  });

  it("GET /api/stocks should return 50 expanded stocks across sectors", async () => {
    const res = await request(app).get("/api/stocks");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("stocks");
    expect(res.body.stocks.length).toBe(50);
    const symbols = res.body.stocks.map((s: any) => s.symbol);
    expect(symbols).toContain("RELIANCE");
    expect(symbols).toContain("TCS");
    expect(symbols).toContain("HDFCBANK");
    expect(symbols).toContain("SUNPHARMA");
    expect(symbols).toContain("MARUTI");
    expect(symbols).toContain("TATASTEEL");
  });

  it("GET /api/stocks?benchmark_only=true should return 10 canonical stocks", async () => {
    const res = await request(app).get("/api/stocks?benchmark_only=true");
    expect(res.status).toBe(200);
    expect(res.body.stocks.length).toBe(10);
  });

  it("GET /api/stocks/sectors should return all 12 Indian industry sectors", async () => {
    const res = await request(app).get("/api/stocks/sectors");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("sectors");
    expect(res.body.sectors.length).toBe(12);
    const sectorNames = res.body.sectors.map((s: any) => s.name);
    expect(sectorNames).toContain("Information Technology");
    expect(sectorNames).toContain("Pharmaceuticals and Healthcare");
    expect(sectorNames).toContain("Automobile and Electric Vehicles");
  });

  it("GET /api/stocks/:symbol should return genuine market quote", async () => {
    const res = await request(app).get("/api/stocks/TCS");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("data");
    expect(res.body.data).toHaveProperty("symbol", "TCS");
    expect(res.body.data).toHaveProperty("currentPrice");
    expect(typeof res.body.data.currentPrice).toBe("number");
  });

  it("POST /api/recommendations should return profile-weighted stock picks", async () => {
    const res = await request(app)
      .post("/api/recommendations")
      .send({
        profile: {
          riskTolerance: "conservative",
          preferredSectors: ["Pharmaceuticals and Healthcare", "FMCG and Consumer Goods"],
          investmentHorizon: "long_term",
          investmentObjective: "wealth_preservation",
          budgetRange: "under_25k",
          sentimentPreference: "balanced",
        },
        top_k: 3,
      });
    expect(res.status).toBe(200);
    expect(res.body.data).toHaveProperty("recommendations");
    expect(res.body.data.recommendations.length).toBeLessThanOrEqual(3);
    expect(res.body.data.recommendations[0]).toHaveProperty("suitabilityScore");
    expect(res.body.data.recommendations[0]).toHaveProperty("matchReasons");
  }, 25000);

  it("GET and POST /api/profile should persist user investor profile", async () => {
    const newProfile = {
      riskTolerance: "aggressive",
      preferredSectors: ["Information Technology", "Automobile and Electric Vehicles"],
      investmentHorizon: "short_term",
      investmentObjective: "capital_growth",
      budgetRange: "100k_to_500k",
      sentimentPreference: "high_sentiment",
    };
    const postRes = await request(app).post("/api/profile").send(newProfile);
    expect(postRes.status).toBe(200);
    expect(postRes.body.profile.riskTolerance).toBe("aggressive");

    const getRes = await request(app).get("/api/profile");
    expect(getRes.status).toBe(200);
    expect(getRes.body.profile.riskTolerance).toBe("aggressive");
  });

  it("GET /api/watchlist should return watchlist items", async () => {
    const res = await request(app).get("/api/watchlist");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("items");
    expect(Array.isArray(res.body.items)).toBe(true);
  });

  it("POST /api/watchlist should add a new stock item", async () => {
    const res = await request(app)
      .post("/api/watchlist")
      .send({
        symbol: "INFY",
        targetPrice: 1250.0,
        notes: "Target buy price",
      });
    expect(res.status).toBe(201);
    expect(res.body).toHaveProperty("success", true);
    expect(res.body.item).toHaveProperty("symbol", "INFY");
  });

  it("DELETE /api/watchlist/:symbol should remove stock", async () => {
    const res = await request(app).delete("/api/watchlist/INFY");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("success", true);
  });

  it("POST /api/predictions/run should validate invalid input", async () => {
    const res = await request(app)
      .post("/api/predictions/run")
      .send({ top_k: 99 }); // invalid top_k > 10
    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty("success", false);
  });

  it("GET /api/predictions/history should return history array", async () => {
    const res = await request(app).get("/api/predictions/history");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("records");
    expect(Array.isArray(res.body.records)).toBe(true);
  });
});
