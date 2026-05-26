import { getAuth } from "@hono/clerk-auth";
import { zValidator } from "@hono/zod-validator";
import { desc, eq, sql } from "drizzle-orm";
import { Hono } from "hono";
import { type SSEStreamingApi, streamSSE } from "hono/streaming";
import { z } from "zod";
import { MAX_INPUT_LIMIT } from "@/lib/constants";
import { db } from "@/lib/db";
import { checks, texts, users } from "@/lib/db/schema";
import { getEvents, streamExists } from "@/lib/redis";
import type { Claim } from "@/lib/store";
import { extractClerkId } from "@/lib/utils";
import {
  executeFactCheckingAgent,
  generateCheckTitle,
  initializeFactCheckSession,
} from "@/server/services/check";
import { after } from "next/server";

const factCheckRequestSchema = z.object({
  content: z.string().trim().min(1).max(MAX_INPUT_LIMIT),
  checkId: z.string().trim().min(1).max(100),
});

const generateTitleRequestSchema = z.object({
  content: z.string().trim().min(10).max(1000),
  checkId: z.string().trim().min(1).max(100),
});

const writeStreamEvent = async (
  stream: SSEStreamingApi,
  eventType: string,
  eventData: unknown
) => {
  await stream.writeSSE({
    event: eventType,
    data: JSON.stringify(eventData),
  });
};

const streamStoredResults = async (
  stream: SSEStreamingApi,
  results: StoredCheckResults
) => {
  await writeStreamEvent(stream, "connection", {
    message: "Connection established",
  });

  await writeStreamEvent(stream, "metadata", {
    user: {
      id: results.users?.id,
      email: results.users?.email,
      image: results.users?.imageUrl,
    },
    text: results.texts?.content,
    title: results.checks.title,
    isPublic: results.checks.isPublic,
    createdAt: results.checks.createdAt,
  });

  if (Array.isArray(results.checks.result)) {
    await writeStreamEvent(stream, "verdicts", {
      claims: results.checks.result,
    });
  }

  await writeStreamEvent(stream, "complete", {
    completed: true,
  });
};

const isTerminalEvent = (eventType: string) =>
  ["complete", "error", "verdicts"].includes(eventType);

const streamLiveEvents = async (stream: SSEStreamingApi, streamId: string) => {
  await writeStreamEvent(stream, "connection", {
    message: "Connection established",
  });

  const existingEvents = await getEvents(streamId);
  let lastProcessedEventId = "0";

  for (const event of existingEvents) {
    const shouldStream = [
      "metadata",
      "sentences",
      "claims",
      "verdicts",
    ].includes(event.event);

    if (shouldStream) {
      await writeStreamEvent(stream, event.event, event.data);
    }

    if (isTerminalEvent(event.event)) {
      return;
    }

    lastProcessedEventId = event.id ?? lastProcessedEventId;
  }

  let streamActive = true;
  stream.onAbort(() => {
    streamActive = false;
  });

  while (streamActive) {
    await new Promise((resolve) => setTimeout(resolve, 1000));

    if (!streamActive) break;

    const currentEvents = await getEvents(streamId);
    const lastKnownIndex = currentEvents.findIndex(
      (event) => event.id === lastProcessedEventId
    );

    const newEvents =
      lastKnownIndex === -1
        ? currentEvents
        : currentEvents.slice(lastKnownIndex + 1);

    for (const event of newEvents) {
      if (!streamActive) break;

      const shouldStream = [
        "metadata",
        "sentences",
        "claims",
        "verdicts",
      ].includes(event.event);

      if (shouldStream) {
        await writeStreamEvent(stream, event.event, event.data);
      }

      lastProcessedEventId = event.id ?? lastProcessedEventId;

      if (isTerminalEvent(event.event)) {
        streamActive = false;
        break;
      }
    }
  }
};

export type StoredCheckResults = {
  checks: typeof checks.$inferSelect;
  users: typeof users.$inferSelect | null;
  texts: typeof texts.$inferSelect | null;
};

const getCompletedCheckResults = async (
  streamId: string
): Promise<StoredCheckResults | null> => {
  const [existingCheck] = await db
    .select()
    .from(checks)
    .where(eq(checks.slug, streamId))
    .leftJoin(texts, eq(checks.textId, texts.id))
    .leftJoin(users, eq(checks.userId, users.id))
    .limit(1);

  if (!existingCheck) return null;

  const isCompleted =
    existingCheck.checks.status === "completed" &&
    !!existingCheck.checks.result;
  return isCompleted ? existingCheck : null;
};

const validateCheckOwnership = async (checkId: string, userId: string) => {
  const [existingCheck] = await db
    .select({ userId: checks.userId, status: checks.status })
    .from(checks)
    .where(eq(checks.slug, checkId))
    .limit(1);

  if (!existingCheck) {
    throw new Error("Check not found");
  }

  if (existingCheck.userId !== extractClerkId(userId)) {
    throw new Error("Unauthorized access to check");
  }

  return existingCheck;
};

export const agentRoute = new Hono()
  .get("/checks", async (context) => {
    const auth = getAuth(context);
    if (!auth?.userId) {
      return context.json({ error: "Unauthorized" }, 401);
    }

    try {
      const userChecks = await db
        .select({
          id: checks.id,
          slug: checks.slug,
          title: checks.title,
          status: checks.status,
          createdAt: checks.createdAt,
          updatedAt: checks.updatedAt,
          completedAt: checks.completedAt,
          textPreview: sql<string>`substring(${texts.content}, 1, 50)`,
        })
        .from(checks)
        .leftJoin(texts, eq(checks.textId, texts.id))
        .where(eq(checks.userId, extractClerkId(auth.userId)))
        .orderBy(desc(checks.updatedAt));

      return context.json({ checks: userChecks });
    } catch (error) {
      console.error("Failed to fetch checks:", error);
      return context.json({ error: "Failed to fetch checks" }, 500);
    }
  })

  .post("/run", zValidator("json", factCheckRequestSchema), async (context) => {
    const auth = getAuth(context);
    if (!auth?.userId) {
      return context.json({ error: "Unauthorized" }, 401);
    }

    const { content, checkId } = context.req.valid("json");

    try {
      const { metadata } = await initializeFactCheckSession({
        content,
        checkId,
        userId: auth.userId,
      });

      after(
        executeFactCheckingAgent({
          streamId: checkId,
          content,
          metadata,
        }).catch((error) =>
          console.error("Background agent processing failed:", error)
        )
      );

      return context.json({ checkId });
    } catch (error) {
      console.error("Failed to initialize fact-check:", error);
      return context.json({ error: "Failed to start fact-check" }, 500);
    }
  })

  .post(
    "/generate-title",
    zValidator("json", generateTitleRequestSchema),
    async (ctx) => {
      const auth = getAuth(ctx);
      if (!auth?.userId) return ctx.json({ error: "Unauthorized" }, 401);

      const { content, checkId } = ctx.req.valid("json");

      try {
        await validateCheckOwnership(checkId, auth.userId);
        const title = await generateCheckTitle(checkId, content);

        return ctx.json({
          title: title || "Title generation failed",
          checkId,
        });
      } catch (error) {
        console.error("Failed to generate title:", error);
        const status =
          error instanceof Error && error.message.includes("not found")
            ? 404
            : error instanceof Error && error.message.includes("Unauthorized")
            ? 403
            : 500;
        return ctx.json(
          {
            error:
              error instanceof Error
                ? error.message
                : "Failed to generate title",
          },
          status
        );
      }
    }
  )

  .get("/stream/:streamId", async (context) => {
    const auth = getAuth(context);
    if (!auth?.userId) {
      return context.json({ error: "Unauthorized" }, 401);
    }

    const streamId = context.req.param("streamId");

    try {
      const storedResults = await getCompletedCheckResults(streamId);

      if (storedResults) {
        return streamSSE(context, (stream) =>
          streamStoredResults(stream, storedResults)
        );
      }

      const streamIsActive = await streamExists(streamId);
      if (!streamIsActive) {
        return context.json({ error: "Stream not found" }, 404);
      }

      return streamSSE(context, (stream) => streamLiveEvents(stream, streamId));
    } catch (error) {
      console.error("Stream connection error:", error);
      return context.json({ error: "Failed to connect to stream" }, 500);
    }
  })

  .patch(
    "/checks/:checkId/public",
    zValidator(
      "json",
      z.object({
        isPublic: z.boolean(),
      })
    ),
    async (context) => {
      const auth = getAuth(context);
      if (!auth?.userId) {
        return context.json({ error: "Unauthorized" }, 401);
      }

      const checkId = context.req.param("checkId");
      const { isPublic } = context.req.valid("json");

      try {
        const existingCheck = await validateCheckOwnership(
          checkId,
          auth.userId
        );

        if (existingCheck.status !== "completed") {
          return context.json(
            { error: "Check must be completed to share" },
            400
          );
        }

        await db
          .update(checks)
          .set({ isPublic })
          .where(eq(checks.slug, checkId));

        return context.json({
          message: isPublic
            ? "Check made public successfully"
            : "Check made private successfully",
          isPublic,
          shareUrl: isPublic ? `/public/${checkId}` : null,
        });
      } catch (error) {
        console.error("Failed to update check visibility:", error);
        const status =
          error instanceof Error && error.message.includes("not found")
            ? 404
            : error instanceof Error && error.message.includes("Unauthorized")
            ? 403
            : 500;
        return context.json(
          {
            error:
              error instanceof Error
                ? error.message
                : "Failed to update check visibility",
          },
          status
        );
      }
    }
  )

  .get("/public/:checkId", async (context) => {
    const checkId = context.req.param("checkId");

    try {
      const [publicCheck] = await db
        .select({
          id: checks.id,
          slug: checks.slug,
          title: checks.title,
          result: checks.result,
          status: checks.status,
          isPublic: checks.isPublic,
          completedAt: checks.completedAt,
          textContent: texts.content,
          createdAt: checks.createdAt,
          user: {
            id: users.id,
            email: users.email,
            image: users.imageUrl,
          },
        })
        .from(checks)
        .leftJoin(texts, eq(checks.textId, texts.id))
        .leftJoin(users, eq(checks.userId, users.id))
        .where(eq(checks.slug, checkId))
        .limit(1);

      if (!publicCheck) return context.json({ error: "Check not found" }, 404);

      if (!publicCheck.isPublic)
        return context.json({ error: "Check is not public" }, 403);

      if (publicCheck.status !== "completed" || !publicCheck.result)
        return context.json({ error: "Check not completed" }, 400);

      return context.json({
        check: publicCheck.result as [string, Claim[]][],
        metadata: {
          user: publicCheck.user,
          text: publicCheck.textContent,
          title: publicCheck.title,
          isPublic: publicCheck.isPublic,
          createdAt: publicCheck.createdAt,
        },
      });
    } catch (error) {
      console.error("Failed to fetch public check:", error);
      return context.json({ error: "Failed to fetch public check" }, 500);
    }
  });
