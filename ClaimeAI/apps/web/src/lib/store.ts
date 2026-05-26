"use client";

import { create } from "zustand";
import { devtools } from "zustand/middleware";
import type { AgentSSEStreamEvent, CheckMetadata, ClaimData } from "@/types";
import { parseSSEEventData, type SSEEvent } from "./event-schema";
import { client } from "./rpc";

export type Claim = ClaimData & {
  sentence: string;
};

interface FactCheckerState {
  text: string;
  isLoading: boolean;
  rawServerEvents: SSEEvent[];
  claims: Map<string, Claim[]>;
  metadata: CheckMetadata | null;
  currentCheckId: string | null;
  hasNoClaims: boolean;
}

interface FactCheckerActions {
  startVerification: (
    text: string,
    checkId: string
  ) => Promise<{ streamId: unknown; checkId: string }>;
  resetState: () => void;
  setIsLoading: (isLoading: boolean) => void;
  setText: (text: string) => void;
  setCurrentCheckId: (checkId: string | null) => void;
  addRawServerEvent: (event: AgentSSEStreamEvent) => void;
  processEventData: (eventData: string) => void;
  connectToStream: (checkId: string) => Promise<void>;
}

type FactCheckerStore = FactCheckerState & FactCheckerActions;

const initialState: FactCheckerState = {
  isLoading: false,
  text: "",
  currentCheckId: null,
  rawServerEvents: [],
  claims: new Map(),
  metadata: null,
  hasNoClaims: false,
};

const createErrorEvent = (
  message: string,
  runId = "local-error"
): AgentSSEStreamEvent => ({
  event: "error",
  data: { message, run_id: runId },
});

const startFactChecking = async (content: string, checkId: string) => {
  try {
    const response = await client.api.agent.run.$post({
      json: { content, checkId },
    });

    if (!response.ok) {
      throw new Error(`Verification failed to start (${response.status})`);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Unable to start fact-check: ${error.message}`);
    }
    throw new Error("Unable to start fact-check");
  }
};

const parseSSEMessage = (
  message: string
): { eventType: string; jsonData: string } | null => {
  const lines = message.split("\n");
  let eventType = "";
  let jsonData = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line.substring(6).trim();
    } else if (line.startsWith("data:")) {
      jsonData = line.substring(5).trim();
    }
  }

  return eventType && jsonData ? { eventType, jsonData } : null;
};

export const useFactCheckerStore = create<FactCheckerStore>()(
  devtools(
    (set, get) => ({
      ...initialState,

      resetState: () =>
        set({ ...initialState, claims: new Map(), metadata: null, hasNoClaims: false }),
      setIsLoading: (isLoading) => set({ isLoading }),
      setCurrentCheckId: (checkId) => set({ currentCheckId: checkId }),
      setText: (text) => set({ text }),

      addRawServerEvent: (event) =>
        set((state) => ({
          rawServerEvents: [...state.rawServerEvents, event],
        })),

      startVerification: async (content: string, checkId: string) => {
        if (!content.trim()) {
          throw new Error("Please provide text to verify");
        }

        get().resetState();
        set({ isLoading: true, currentCheckId: checkId });

        try {
          const result = await startFactChecking(content, checkId);
          return { streamId: result, checkId };
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "Verification failed";
          get().addRawServerEvent(createErrorEvent(message));
          set({ isLoading: false });
          throw error;
        }
      },

      processEventData: (eventData) => {
        try {
          const event = parseSSEEventData(eventData);
          get().addRawServerEvent(event);

          if (event.event === "metadata") set({ metadata: event.data });

          if (event.event === "no-claims") set({ hasNoClaims: true });

          const claimsEvents = ["sentences", "claims", "verdicts"];
          if (claimsEvents.includes(event.event)) {
            const { claims } = event.data as { claims: [string, Claim[]][] };
            set((state) => ({
              claims: new Map([...state.claims, ...claims]),
            }));
          }
        } catch (error) {
          console.error("Failed to process event:", error);
        }
      },

      connectToStream: async (checkId: string) => {
        const {
          currentCheckId,
          resetState,
          setCurrentCheckId,
          setIsLoading,
          addRawServerEvent,
          processEventData,
        } = get();

        if (currentCheckId !== checkId) {
          resetState();
          setCurrentCheckId(checkId);
          setIsLoading(true);
        }

        try {
          // For streaming endpoints, we still use fetch as RPC clients may not handle SSE properly
          const response = await fetch(`/api/agent/stream/${checkId}`);

          if (!response.ok) {
            throw new Error(`Connection failed (${response.status})`);
          }

          if (!response.body) {
            throw new Error("No response received");
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          try {
            while (true) {
              const { done, value } = await reader.read();

              if (done) {
                setIsLoading(false);
                break;
              }

              buffer += decoder.decode(value, { stream: true });
              const messages = buffer.split("\n\n");

              for (let i = 0; i < messages.length - 1; i++) {
                const message = messages[i].trim();
                if (!message) continue;

                try {
                  const parsed = parseSSEMessage(message);
                  if (!parsed) continue;

                  const { eventType, jsonData } = parsed;
                  const eventData = {
                    event: eventType,
                    data: JSON.parse(jsonData),
                  };

                  processEventData(JSON.stringify(eventData));

                  if (["complete", "error", "verdicts", "no-claims"].includes(eventType)) {
                    setIsLoading(false);
                  }
                } catch (error) {
                  console.error("Message parsing failed:", error);
                }
              }

              const lastDelimiterPos = buffer.lastIndexOf("\n\n");
              buffer =
                lastDelimiterPos !== -1
                  ? buffer.substring(lastDelimiterPos + 2)
                  : buffer;
            }
          } finally {
            reader.releaseLock();
          }
        } catch (error) {
          const message =
            error instanceof Error ? error.message : "Connection failed";
          addRawServerEvent(createErrorEvent(message));
          setIsLoading(false);
        }
      },
    }),
    { name: "claimeai-store" }
  )
);
