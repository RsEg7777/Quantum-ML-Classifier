import { NextRequest, NextResponse } from "next/server";

import { getRequestUser } from "@/lib/server/auth";
import { getUpload } from "@/lib/server/upload-store";

type Params = {
  params: Promise<{ id: string }>;
};

export async function GET(request: NextRequest, context: Params) {
  const { id } = await context.params;
  const upload = await getUpload(id);

  if (!upload) {
    return NextResponse.json({ error: "Upload not found" }, { status: 404 });
  }

  const token = request.nextUrl.searchParams.get("token");
  const user = getRequestUser(request);

  if (token !== upload.token && !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return new NextResponse(upload.content, {
    status: 200,
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": `inline; filename="${upload.filename}"`,
      "Cache-Control": "no-store",
    },
  });
}
