import { clerkMiddleware } from "@hono/clerk-auth";
import { Hono } from "hono";
import { handle } from "hono/vercel";
import { env } from "@/env";
import { agentRoute } from "@/server/routes/agent";

export const dynamic = "force-dynamic";

const app = new Hono().basePath("/api");

app.use(
  "*",
  clerkMiddleware({
    publishableKey: env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY,
    secretKey: env.CLERK_SECRET_KEY,
  })
);
const routes = app
  .get("/health", (c) => {
    return c.json({
      message: "OK",
    });
  })
  .route("/agent", agentRoute);

type AppType = typeof routes;

const GET = handle(app);
const POST = handle(app);
const PATCH = handle(app);

export { GET, POST, PATCH };
export type { AppType };
