import { Client } from "@langchain/langgraph-sdk";
import { env } from "@/env";

export const client = new Client({
  apiUrl: env.LANGGRAPH_API_URL,
  apiKey: env.LANGSMITH_API_KEY,
});
