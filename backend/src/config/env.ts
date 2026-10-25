import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const envSchema = z.object({
  PORT: z.string().default("5000").transform((val) => parseInt(val, 10)),
  NODE_ENV: z.enum(["development", "production", "test"]).default("development"),
  PYTHON_ML_SERVICE_URL: z.string().url().default("http://127.0.0.1:8000"),
  DATABASE_URL: z.string().optional().default("postgresql://postgres:postgres@localhost:5432/spcm_intelligence?schema=public"),
  CORS_ORIGIN: z.string().default("http://localhost:5173,http://127.0.0.1:5173"),
});

export type EnvConfig = z.infer<typeof envSchema>;

let parsedEnv: EnvConfig;
try {
  parsedEnv = envSchema.parse(process.env);
} catch (error) {
  console.warn("[Config] Invalid or missing environment variables; using sensible defaults:", error);
  parsedEnv = {
    PORT: 5000,
    NODE_ENV: "development",
    PYTHON_ML_SERVICE_URL: "http://127.0.0.1:8000",
    DATABASE_URL: "postgresql://postgres:postgres@localhost:5432/spcm_intelligence?schema=public",
    CORS_ORIGIN: "http://localhost:5173,http://127.0.0.1:5173",
  };
}

export const env = parsedEnv;
