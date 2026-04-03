import { NextRequest, NextResponse } from "next/server";
import { ZodError, z } from "zod";

import { updateJobProgress } from "@/lib/server/job-store";

const ProgressPayloadSchema = z.object({
  id: z.string().min(8),
  progress: z.object({
    percent: z.number().int().min(0).max(100),
    stage: z.string().min(1),
    message: z.string().optional(),
    steps: z
      .array(
        z.object({
          label: z.string().min(1),
          state: z.enum(["done", "active", "pending", "error"]),
        })
      )
      .max(10)
      .optional(),
  }),
});

function isAuthorized(request: NextRequest): boolean {
  const secret = process.env.WORKER_WEBHOOK_SECRET;
  if (!secret) {
    return false;
  }

  const token = request.headers.get("x-dispatch-secret");
  return token === secret;
}

export async function POST(request: NextRequest) {
  if (!process.env.WORKER_WEBHOOK_SECRET) {
    return NextResponse.json(
      { error: "Server dispatch secret is not configured" },
      { status: 500 }
    );
  }

  if (!isAuthorized(request)) {
    return NextResponse.json({ error: "Unauthorized progress update" }, { status: 401 });
  }

  try {
    const payload = ProgressPayloadSchema.parse(await request.json());
    await updateJobProgress(payload.id, payload.progress);
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
