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

  it("GET /api/stocks should return 10 canonical stocks", async () => {
    const res = await request(app).get("/api/stocks");
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("stocks");
    expect(res.body.stocks.length).toBe(10);
    const symbols = res.body.stocks.map((s: any) => s.symbol);
    expect(symbols).toContain("RELIANCE");
    expect(symbols).toContain("TCS");
    expect(symbols).toContain("HDFCBANK");
    expect(symbols).toContain("INFY");
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
