import { createEnv } from "@t3-oss/env-core";
import { z } from "zod";

export const env = createEnv({
  server: {
    LANGGRAPH_API_URL: z.url().optional().default("http://localhost:2024"),
    LANGSMITH_API_KEY: z.string().min(1).optional(),
    OPENAI_API_KEY: z.string().min(1).startsWith("sk-"),
    CLERK_SECRET_KEY: z.string().min(1).startsWith("sk_"),
    CLERK_WEBHOOK_SECRET: z.string().min(1),
    DATABASE_URL: z.string().min(1),
    UPSTASH_REDIS_REST_URL: z.url().min(1),
    UPSTASH_REDIS_REST_TOKEN: z.string().min(1),
    NODE_ENV: z
      .enum(["development", "production"])
      .optional()
      .default("development"),
  },

  /**
   * The prefix that client-side variables must have. This is enforced both at
   * a type-level and at runtime.
   */
  clientPrefix: "NEXT_PUBLIC_",

  client: {
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: z.string().min(1).startsWith("pk_"),
    NEXT_PUBLIC_APP_URL: z.url().default("http://localhost:3000"),
  },

  /**
   * What object holds the environment variables at runtime. This is usually
   * `process.env` or `import.meta.env`.
   */
  runtimeEnv: {
    ...process.env,
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY:
      process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY,
    NEXT_PUBLIC_APP_URL: process.env.NEXT_PUBLIC_APP_URL,
  },

  /**
   * By default, this library will feed the environment variables directly to
   * the Zod validator.
   *
   * This means that if you have an empty string for a value that is supposed
   * to be a number (e.g. `PORT=` in a ".env" file), Zod will incorrectly flag
   * it as a type mismatch violation. Additionally, if you have an empty string
   * for a value that is supposed to be a string with a default value (e.g.
   * `DOMAIN=` in an ".env" file), the default value will never be applied.
   *
   * In order to solve these issues, we recommend that all new projects
   * explicitly specify this option as true.
   */
  emptyStringAsUndefined: true,
});
