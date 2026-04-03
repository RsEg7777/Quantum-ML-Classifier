import { NextRequest, NextResponse } from "next/server";

import { getRequestUser } from "@/lib/server/auth";
import { listDatasets } from "@/lib/server/job-store";

export async function GET(request: NextRequest) {
  const user = getRequestUser(request);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({ datasets: listDatasets() });
}
