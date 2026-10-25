import { describe, it, expect } from "vitest";
import { DEMO_PREDICTIONS } from "../services/api.js";
import { StockPriceChart } from "../components/charts/StockPriceChart.js";
import { ProbabilityBarChart } from "../components/charts/ProbabilityBarChart.js";

describe("Frontend Service and Model Tests", () => {
  it("DEMO_PREDICTIONS should contain canonical Indian stocks", () => {
    expect(DEMO_PREDICTIONS.is_demo_data).toBe(true);
    expect(DEMO_PREDICTIONS.recommended_stocks.length).toBeGreaterThan(0);
    const stocks = DEMO_PREDICTIONS.predictions.map((p) => p.stock);
    expect(stocks).toContain("RELIANCE");
    expect(stocks).toContain("TCS");
    expect(stocks).toContain("SBIN");
  });

  it("DEMO_PREDICTIONS should have valid probabilities in [0, 1]", () => {
    for (const pred of DEMO_PREDICTIONS.predictions) {
      expect(pred.probability).toBeGreaterThanOrEqual(0);
      expect(pred.probability).toBeLessThanOrEqual(1);
      expect(pred.rank).toBeGreaterThanOrEqual(1);
    }
  });

  it("DEMO_PREDICTIONS should have matching recommended flag for top_k", () => {
    const recommended = DEMO_PREDICTIONS.predictions.filter((p) => p.is_recommended);
    expect(recommended.length).toBe(DEMO_PREDICTIONS.top_k);
  });

  it("DEMO_PREDICTIONS should include valid market_metadata and news_metadata schema", () => {
    expect(DEMO_PREDICTIONS.market_metadata).toBeDefined();
    expect(DEMO_PREDICTIONS.market_metadata.feed_type).toBeDefined();
    expect(DEMO_PREDICTIONS.news_metadata).toBeDefined();
    expect(DEMO_PREDICTIONS.model_metadata).toBeDefined();
  });
});

describe("Chart Components Export Tests", () => {
  it("StockPriceChart and ProbabilityBarChart components should be defined functions", () => {
    expect(typeof StockPriceChart).toBe("function");
    expect(typeof ProbabilityBarChart).toBe("function");
  });
});

