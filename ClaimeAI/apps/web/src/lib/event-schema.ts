import { z } from "zod";

export const errorEventSchema = z.object({
  event: z.literal("error"),
  data: z.object({
    message: z.string(),
    run_id: z.string(),
  }),
});

export const updatesEventSchema = z.object({
  event: z.string(),
  data: z.any(),
});

export type SSEEvent = z.infer<typeof updatesEventSchema>;

export const parseSSEEventData = (eventData: string): SSEEvent => {
  try {
    const parsed = JSON.parse(eventData);

    if (!parsed || typeof parsed !== "object") {
      throw new Error("Invalid event data: not an object");
    }

    if (!parsed.event || typeof parsed.event !== "string") {
      throw new Error(
        "Invalid event data: missing or invalid 'event' property"
      );
    }

    if (parsed.data === undefined) {
      throw new Error("Invalid event data: missing 'data' property");
    }

    return updatesEventSchema.parse(parsed);
  } catch (error) {
    if (error instanceof Error) {
      console.error(`Failed to parse SSE event data: ${error.message}`, error);
      console.error("Problematic event data:", eventData);
    } else {
      console.error("Unknown error parsing SSE event data");
    }
    throw error;
  }
};
