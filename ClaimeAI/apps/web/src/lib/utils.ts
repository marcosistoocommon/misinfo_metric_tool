import type { UserJSON } from "@clerk/nextjs/server";
import { type ClassValue, clsx } from "clsx";
import { nanoid } from "nanoid";
import { twMerge } from "tailwind-merge";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

/**
 * Extracts the unique identifier from a Clerk user ID
 */
export const extractClerkId = (clerkUserId: string): string =>
  clerkUserId.startsWith("user_") ? clerkUserId.slice(5) : clerkUserId;

export const extractDomain = (url: string): string => {
  try {
    const hostname = new URL(url).hostname;
    return hostname.startsWith("www.") ? hostname.substring(4) : hostname;
  } catch (_e) {
    return url;
  }
};

export const generateCheckId = (text: string): string => {
  const firstSentence = text.split(/[.!?]/)[0] || text;
  const truncated =
    firstSentence.length > 60
      ? firstSentence.substring(0, 60).trim()
      : firstSentence.trim();

  const slug = truncated
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "") // Remove special characters
    .replace(/[\s-]+/g, "-") // Replace spaces and hyphens with a single hyphen
    .replace(/^-|-$/g, ""); // Remove leading/trailing hyphens

  const hash = nanoid(8);

  const finalSlug = slug || "check";

  return `${finalSlug}-${hash}`;
};

export const getPrimaryEmail = (user: UserJSON): string | undefined =>
  user.email_addresses.find(
    (email) => email.id === user.primary_email_address_id
  )?.email_address;
