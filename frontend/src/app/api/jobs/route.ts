import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";

import { getRequestUser } from "@/lib/server/auth";
import { createJob, listJobs } from "@/lib/server/job-store";

export async function GET(request: NextRequest) {
  const user = getRequestUser(request);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({ jobs: await listJobs() });
}

export async function POST(request: NextRequest) {
  const user = getRequestUser(request);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const payload = await request.json();
    const job = await createJob(payload);
    return NextResponse.json({ job }, { status: 201 });
  } catch (error) {
    if (error instanceof ZodError) {
      return NextResponse.json(
        { error: "Invalid payload", issues: error.issues },
        { status: 400 }
      );
    }

    return NextResponse.json(
      { error: "Unexpected error while creating job." },
      { status: 500 }
    );
  }
}
