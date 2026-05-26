import type { UserJSON, WebhookEvent } from "@clerk/nextjs/server";
import { headers } from "next/headers";
import { NextResponse } from "next/server";
import { Webhook } from "svix";
import { env } from "@/env";
import { db } from "@/lib/db";
import { users } from "@/lib/db/schema/user";
import { extractClerkId, getPrimaryEmail } from "@/lib/utils";

const handleUserCreated = async (data: UserJSON) => {
  const id = extractClerkId(data.id);
  const email = getPrimaryEmail(data);

  if (!email) return new Response("User has no primary email", { status: 400 });

  await db.insert(users).values({
    id,
    email,
    imageUrl: data.image_url,
  });

  return new Response("User created", { status: 201 });
};

export const POST = async (request: Request): Promise<Response> => {
  if (!env.CLERK_WEBHOOK_SECRET) {
    return NextResponse.json({ message: "Not configured", ok: false });
  }

  const headerPayload = await headers();
  const svixId = headerPayload.get("svix-id");
  const svixTimestamp = headerPayload.get("svix-timestamp");
  const svixSignature = headerPayload.get("svix-signature");

  if (!(svixId && svixTimestamp && svixSignature)) {
    return new Response("Error occurred: no svix headers", { status: 400 });
  }

  const payload = (await request.json()) as object;
  const body = JSON.stringify(payload);

  const webhook = new Webhook(env.CLERK_WEBHOOK_SECRET);

  let event: WebhookEvent | undefined;

  try {
    event = webhook.verify(body, {
      "svix-id": svixId,
      "svix-timestamp": svixTimestamp,
      "svix-signature": svixSignature,
    }) as WebhookEvent;
  } catch (error) {
    console.error("Error verifying webhook:", { error });
    return new Response("Error occured", { status: 400 });
  }

  const { id } = event.data;
  const eventType = event.type;

  console.log("Webhook", { id, eventType, body });

  let response: Response = new Response("", { status: 201 });

  switch (eventType) {
    case "user.created": {
      response = await handleUserCreated(event.data);
      break;
    }
    default: {
      break;
    }
  }

  return response;
};
