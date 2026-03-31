import { NextResponse } from "next/server";

import { listDatasets } from "@/lib/server/job-store";

export async function GET() {
  return NextResponse.json({ datasets: listDatasets() });
}
