import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";
import { eq } from "drizzle-orm";
import { z } from "zod";

import { db } from "@/lib/db";
import {
  createCheck,
  findOrCreateText,
  getUserById,
  updateCheckResult,
  updateCheckStatus,
} from "@/lib/db/operations";
import { checks } from "@/lib/db/schema";
import { client } from "@/lib/langgraph";
import {
  addEvent,
  completeStream,
  createStream,
  failStream,
  getEvents,
} from "@/lib/redis";
import type {
  CheckMetadata,
  ClaimData,
  ClaimsEventData,
  ErrorEventData,
  ExecuteAgentParams,
  InitializeSessionParams,
  SanitizedEvent,
} from "@/types";

const agentEventSchema = z.object({
  event: z.string(),
  data: z.unknown(),
});

const sanitizeEventData = (
  event: string,
  // biome-ignore lint/suspicious/noExplicitAny: Event data from agent has dynamic structure
  data: any
): SanitizedEvent[] => {
  const events: SanitizedEvent[] = [];

  if (!data || typeof data !== "object") return events;

  if (
    event.startsWith("updates|extract_claims:") &&
    data.sentence_splitter?.contextual_sentences
  ) {
    const claimsMap = new Map<string, ClaimData[]>();

    for (const sentence of data.sentence_splitter.contextual_sentences) {
      claimsMap.set(sentence.original_sentence, []);
    }

    events.push({
      event: "sentences",
      data: { claims: Array.from(claimsMap.entries()) },
    });
  }

  if (event === "updates" && data.extract_claims?.extracted_claims) {
    const claimsMap = new Map<string, ClaimData[]>();

    for (const claim of data.extract_claims.extracted_claims) {
      const sentence = claim.original_sentence;
      const existingClaims = claimsMap.get(sentence) || [];

      claimsMap.set(sentence, [
        ...existingClaims,
        { text: claim.claim_text, status: "pending" },
      ]);
    }

    events.push({
      event: "claims",
      data: { claims: Array.from(claimsMap.entries()) },
    });
  }

  if (event === "updates" && data.generate_report_node?.final_report) {
    const claimsMap = new Map<string, ClaimData[]>();

    for (const claim of data.generate_report_node.final_report
      .verified_claims) {
      const sentence = claim.original_sentence;
      const existingClaims = claimsMap.get(sentence) || [];

      claimsMap.set(sentence, [
        ...existingClaims,
        {
          status: "verified",
          text: claim.claim_text,
          result: claim.result as "Supported" | "Refuted",
          reasoning: claim.reasoning,
          sources: claim.sources || [],
        },
      ]);
    }

    if (claimsMap.size > 0) {
      events.push({
        event: "verdicts",
        data: { claims: Array.from(claimsMap.entries()) },
      });
    }
  }

  return events;
};

const processAgentEvent = async (
  streamId: string,
  rawEvent: unknown
): Promise<void> => {
  const parseResult = agentEventSchema.safeParse(rawEvent);

  if (!parseResult.success) {
    console.error(
      `[DEBUG] Invalid agent event structure for ${streamId}:`,
      parseResult.error
    );
    const errorData: ErrorEventData = {
      message: "Server received malformed event data",
      run_id: "validation-error",
    };
    await addEvent(streamId, "error", errorData);
    return;
  }

  const sanitizedEvents = sanitizeEventData(
    parseResult.data.event,
    parseResult.data.data
  );

  for (const sanitizedEvent of sanitizedEvents) {
    await addEvent(streamId, sanitizedEvent.event, sanitizedEvent.data);
  }
};

const persistAgentResults = async (streamId: string): Promise<void> => {
  try {
    const agentEvents = await getEvents(streamId);

    const result = agentEvents.find((event) => event.event === "verdicts");

    if (!result?.data) {
      await addEvent(streamId, "no-claims", {
        message: "No factual claims found in the provided text",
      });
      await updateCheckResult(streamId, []);
      await updateCheckStatus(streamId, "no_claims");
      return;
    }

    await updateCheckResult(streamId, (result.data as ClaimsEventData).claims);
    await updateCheckStatus(streamId, "completed");
  } catch (error) {
    console.error("Failed to persist agent results:", error);
    await updateCheckStatus(streamId, "failed");
  }
};

const handleProcessingError = async (
  streamId: string,
  error: unknown
): Promise<void> => {
  const errorMessage = error instanceof Error ? error.message : "Unknown error";

  const errorData: ErrorEventData = {
    message: errorMessage,
    run_id: "server-error",
  };

  await addEvent(streamId, "error", errorData);

  await failStream(streamId, errorMessage);

  await updateCheckStatus(streamId, "failed");
};

const executeAgentWorkflow = async (
  streamId: string,
  content: string
): Promise<void> => {
  const thread = await client.threads.create();

  const runStream = client.runs.stream(thread.thread_id, "fact_checker", {
    input: { answer: content },
    streamMode: ["updates"],
  });

  for await (const event of runStream) {
    if (event.event === "error") {
      const errorData: ErrorEventData = {
        message: event.data.message || "Agent workflow error",
        run_id: "agent-error",
      };
      await addEvent(streamId, "error", errorData);
      throw new Error(`Agent workflow failed: ${JSON.stringify(event.data)}`);
    }
    await processAgentEvent(streamId, event);
  }
};

export const generateCheckTitle = async (
  checkId: string,
  content: string
): Promise<string | null> => {
  try {
    const { text: generatedTitle } = await generateText({
      model: openai("gpt-4.1-nano"),
      prompt: `Generate a concise, descriptive title (max 5 words) for this fact-checking content. The title should summarize the main claim or topic being fact-checked.

      Examples:

      Content: "The COVID-19 vaccine contains microchips that can track your location and was developed by Bill Gates to control the population."
      Title: COVID-19 Vaccine Microchip and Population Control Claims

      Content: "Drinking 8 glasses of water daily is essential for good health and was recommended by doctors for decades."
      Title: Daily Water Intake Recommendation Claims

      Content: "Climate change is a hoax created by scientists to get more funding, and global temperatures haven't actually risen in the past decade."
      Title: Climate Change Hoax and Temperature Data Claims

      Content: "Eating carrots improves your eyesight significantly and can help you see in the dark, which is why pilots ate them during World War II."
      Title: Carrots and Night Vision Enhancement Claims

      Content: "The moon landing in 1969 was staged by NASA in a Hollywood studio and never actually happened."
      Title: Moon Landing Authenticity Claims

      Now generate a title for:
      "${content}"

      Title:`,
      maxTokens: 100,
      temperature: 0.3,
    });

    const cleanTitle = generatedTitle.trim().replace(/^["']|["']$/g, "");

    await db
      .update(checks)
      .set({ title: cleanTitle, updatedAt: new Date() })
      .where(eq(checks.slug, checkId));

    return cleanTitle;
  } catch (error) {
    console.error("Failed to generate title:", error);
    return null;
  }
};

export const initializeFactCheckSession = async ({
  content,
  checkId,
  userId,
}: InitializeSessionParams) => {
  const [text, user] = await Promise.all([
    findOrCreateText(content),
    getUserById(userId),
  ]);

  if (!user) {
    throw new Error("User not found");
  }

  const check = await createCheck({
    checkId,
    userId,
    textId: text.id,
  });

  const metadata: CheckMetadata = {
    user: {
      id: user.id,
      email: user.email,
      image: user.imageUrl,
    },
    isPublic: check.isPublic,
    text: content,
    title: check.title,
    createdAt: check.createdAt.toISOString(),
  };

  return { text, check, metadata };
};

export const executeFactCheckingAgent = async ({
  streamId,
  content,
  metadata,
}: ExecuteAgentParams): Promise<void> => {
  try {
    await createStream(streamId);

    await addEvent(streamId, "connected", {
      message: "Connected to SSE",
      streamId,
    });

    await addEvent(streamId, "metadata", metadata);

    await executeAgentWorkflow(streamId, content);

    await completeStream(streamId);

    await persistAgentResults(streamId);
  } catch (error) {
    await handleProcessingError(streamId, error);
  }
};
