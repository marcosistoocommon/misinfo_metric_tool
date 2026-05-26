import { createHash } from "node:crypto";
import { eq } from "drizzle-orm";
import { extractClerkId } from "../utils";
import { db } from ".";
import { checks, texts, users } from "./schema";

const generateContentHash = (content: string): string => {
  return createHash("sha256").update(content.trim()).digest("hex");
};

const calculateWordCount = (content: string): number => {
  return content.trim().split(/\s+/).length;
};

export const findOrCreateText = async (content: string) => {
  const contentHash = generateContentHash(content);
  const wordCount = calculateWordCount(content).toString();

  const existingText = await db
    .select()
    .from(texts)
    .where(eq(texts.hash, contentHash))
    .limit(1);

  if (existingText.length > 0) {
    return existingText[0];
  }

  const [newText] = await db
    .insert(texts)
    .values({
      hash: contentHash,
      content,
      wordCount,
    })
    .returning();

  return newText;
};

export const getUserById = async (userId: string) => {
  const clerkId = extractClerkId(userId);
  const [user] = await db
    .select()
    .from(users)
    .where(eq(users.id, clerkId))
    .limit(1);

  return user;
};

type CreateCheckParams = {
  checkId: string;
  userId: string;
  textId: string;
};

export const createCheck = async ({
  checkId,
  userId,
  textId,
}: CreateCheckParams) => {
  const [newCheck] = await db
    .insert(checks)
    .values({
      slug: checkId,
      userId: extractClerkId(userId),
      textId,
      status: "pending",
    })
    .returning();

  return newCheck;
};

export const updateCheckResult = async (
  checkId: string,
  result: unknown
): Promise<void> => {
  await db
    .update(checks)
    .set({
      result,
      status: "completed",
      completedAt: new Date(),
    })
    .where(eq(checks.slug, checkId));
};

export const updateCheckStatus = async (
  checkId: string,
  status: "completed" | "failed" | "no_claims"
): Promise<void> => {
  await db
    .update(checks)
    .set({
      status,
      completedAt: new Date(),
    })
    .where(eq(checks.slug, checkId));
};
