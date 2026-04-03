import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { sql } from "@vercel/postgres";
import { put } from "@vercel/blob";

type StoredUpload = {
  id: string;
  filename: string;
  mimeType: string;
  content: string;
  token: string;
  createdAt: string;
};

const blobEnabled = Boolean(process.env.BLOB_READ_WRITE_TOKEN);
const postgresEnabled = Boolean(process.env.POSTGRES_URL);
let schemaInitialized = false;
const localDataDir = path.join(process.cwd(), ".local-data");
const localUploadsPath = path.join(localDataDir, "uploads.json");

function nowIso(): string {
  return new Date().toISOString();
}

async function ensureSchema(): Promise<void> {
  if ((!postgresEnabled && !blobEnabled) || schemaInitialized) {
    return;
  }

  await sql`
    CREATE TABLE IF NOT EXISTS uploads (
      id TEXT PRIMARY KEY,
      filename TEXT NOT NULL,
      mime_type TEXT NOT NULL,
      content TEXT,
      blob_url TEXT,
      access_token TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL
    )
  `;

  schemaInitialized = true;
}

async function loadLocalUploads(): Promise<StoredUpload[]> {
  try {
    const raw = await readFile(localUploadsPath, "utf8");
    return JSON.parse(raw) as StoredUpload[];
  } catch {
    return [];
  }
}

async function saveLocalUploads(items: StoredUpload[]): Promise<void> {
  await mkdir(localDataDir, { recursive: true });
  await writeFile(localUploadsPath, JSON.stringify(items, null, 2), "utf8");
}

export async function createUpload(input: {
  filename: string;
  mimeType: string;
  content: string;
}): Promise<StoredUpload> {
  const upload: StoredUpload = {
    id: randomUUID(),
    filename: input.filename,
    mimeType: input.mimeType || "text/csv",
    content: "", // Will be stored separately in Blob if enabled
    token: randomUUID(),
    createdAt: nowIso(),
  };

  // Priority 1: Store in Vercel Blob (production)
  if (blobEnabled) {
    await ensureSchema();
    const blobKey = `uploads/${upload.id}/${input.filename}`;

    const blob = await put(blobKey, input.content, {
      contentType: input.mimeType || "text/csv",
      access: "public",
    });

    // Store metadata in Postgres for retrieval
    await sql`
      INSERT INTO uploads (id, filename, mime_type, blob_url, access_token, created_at)
      VALUES (
        ${upload.id},
        ${upload.filename},
        ${upload.mimeType},
        ${blob.url},
        ${upload.token},
        ${upload.createdAt}
      )
    `;

    return {
      ...upload,
      content: blob.url, // Return public URL for worker
    };
  }

  // Priority 2: Store in PostgreSQL
  if (postgresEnabled) {
    await ensureSchema();
    await sql`
      INSERT INTO uploads (id, filename, mime_type, content, access_token, created_at)
      VALUES (
        ${upload.id},
        ${upload.filename},
        ${upload.mimeType},
        ${input.content},
        ${upload.token},
        ${upload.createdAt}
      )
    `;

    return {
      ...upload,
      content: input.content,
    };
  }

  // Priority 3: Store in local JSON (dev fallback)
  const current = await loadLocalUploads();
  const next = current.filter((item) => item.id !== upload.id);
  const localUpload = {
    ...upload,
    content: input.content,
  };
  next.push(localUpload);
  await saveLocalUploads(next);

  return localUpload;
}

export async function getUpload(id: string): Promise<StoredUpload | null> {
  // Priority 1: Retrieve from Blob metadata (in Postgres)
  if (blobEnabled) {
    await ensureSchema();
    const result = await sql<{
      id: string;
      filename: string;
      mime_type: string;
      blob_url: string;
      access_token: string;
      created_at: Date | string;
    }>`
      SELECT id, filename, mime_type, blob_url, access_token, created_at
      FROM uploads
      WHERE id = ${id}
      LIMIT 1
    `;

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      filename: row.filename,
      mimeType: row.mime_type,
      content: row.blob_url || "", // Store blob URL as content for worker
      token: row.access_token,
      createdAt: new Date(row.created_at).toISOString(),
    };
  }

  // Priority 2: Retrieve from PostgreSQL
  if (postgresEnabled) {
    await ensureSchema();
    const result = await sql<{
      id: string;
      filename: string;
      mime_type: string;
      content: string;
      access_token: string;
      created_at: Date | string;
    }>`
      SELECT id, filename, mime_type, content, access_token, created_at
      FROM uploads
      WHERE id = ${id}
      LIMIT 1
    `;

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      filename: row.filename,
      mimeType: row.mime_type,
      content: row.content,
      token: row.access_token,
      createdAt: new Date(row.created_at).toISOString(),
    };
  }

  // Priority 3: Retrieve from local JSON
  const current = await loadLocalUploads();
  return current.find((item) => item.id === id) ?? null;
}
