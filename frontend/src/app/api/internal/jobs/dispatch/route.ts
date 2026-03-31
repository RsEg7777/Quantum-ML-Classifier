import { NextRequest, NextResponse } from "next/server";
import { ZodError, z } from "zod";

import { executeQueuedJob } from "@/lib/server/job-store";

const DispatchPayloadSchema = z.object({
  id: z.string().min(8),
  input: z.unknown(),
});

function isAuthorized(request: NextRequest): boolean {
  const secret = process.env.WORKER_WEBHOOK_SECRET;
  if (!secret) {
    return process.env.NODE_ENV !== "production";
  }

  const token = request.headers.get("x-dispatch-secret");
  return token === secret;
}

export async function POST(request: NextRequest) {
  if (!isAuthorized(request)) {
    return NextResponse.json({ error: "Unauthorized dispatch" }, { status: 401 });
  }

  try {
    const payload = DispatchPayloadSchema.parse(await request.json());
    await executeQueuedJob(payload.id, payload.input);
    return NextResponse.json({ status: "accepted" }, { status: 202 });
  } catch (error) {
    if (error instanceof ZodError) {
      return NextResponse.json(
        { error: "Invalid payload", issues: error.issues },
        { status: 400 }
      );
    }

    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
