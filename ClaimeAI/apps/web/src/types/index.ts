export type EventType = "metadata" | "updates" | string;

export interface AgentEvent<T extends EventType, D> {
  event: T;
  data: D;
}

export interface CheckUser {
  id: string;
  email: string;
  image: string | null;
}

export interface CheckMetadata {
  user: CheckUser;
  text: string;
  title: string | null;
  isPublic: boolean;
  createdAt: string;
}

export interface MetadataEventData {
  run_id: string;
  attempt?: number;
}

export type AgentMetadataEvent = AgentEvent<"metadata", MetadataEventData>;

// biome-ignore lint/suspicious/noExplicitAny: <explanation>
export type AgentUpdatesDataContent = Record<string, any>;

export type AgentUpdatesEvent = AgentEvent<string, AgentUpdatesDataContent>;

export type AgentSSEStreamEvent = AgentMetadataEvent | AgentUpdatesEvent;

// Check service types

export interface InitializeSessionParams {
  content: string;
  checkId: string;
  userId: string;
}

export interface ExecuteAgentParams {
  streamId: string;
  content: string;
  metadata: CheckMetadata;
}

export interface ClaimSource {
  url: string;
  title: string;
  is_influential?: boolean;
}

export type ClaimVerdict = "Supported" | "Refuted";

export type ClaimStatus = "pending" | "verified";

export interface ClaimData {
  text: string;
  status: ClaimStatus;
  result?: ClaimVerdict;
  reasoning?: string;
  sources?: ClaimSource[];
}

export interface SanitizedEvent {
  event: string;
  data: unknown;
}

export interface ClaimsEventData {
  claims: [string, ClaimData[]][];
}

export interface ErrorEventData {
  message: string;
  run_id: string;
}
