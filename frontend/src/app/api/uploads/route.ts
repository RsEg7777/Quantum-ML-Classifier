import { NextRequest, NextResponse } from "next/server";

import { getRequestUser } from "@/lib/server/auth";
import { createUpload } from "@/lib/server/upload-store";

const MAX_BYTES = 2 * 1024 * 1024;

export async function POST(request: NextRequest) {
  const user = getRequestUser(request);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json(
      { error: "Invalid multipart form data" },
      { status: 400 }
    );
  }

  const file = form.get("file");

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "No file provided" }, { status: 400 });
  }

  if (file.size === 0 || file.size > MAX_BYTES) {
    return NextResponse.json(
      { error: "CSV file must be between 1 byte and 2 MB" },
      { status: 400 }
    );
  }

  const content = await file.text();
  const header = content.split(/\r?\n/)[0] ?? "";
  if (!header.includes(",")) {
    return NextResponse.json(
      { error: "CSV must include comma-separated columns and header row" },
      { status: 400 }
    );
  }

  const upload = await createUpload({
    filename: file.name || "dataset.csv",
    mimeType: file.type || "text/csv",
    content,
  });

  const appBaseUrl =
    process.env.APP_BASE_URL ??
    process.env.NEXT_PUBLIC_APP_URL ??
    "http://127.0.0.1:3000";

  const csvBlobUrl = `${appBaseUrl}/api/uploads/${upload.id}?token=${upload.token}`;

  return NextResponse.json({
    uploadId: upload.id,
    csvBlobUrl,
    filename: upload.filename,
  });
}
