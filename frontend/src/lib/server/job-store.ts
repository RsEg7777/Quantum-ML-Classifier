import { randomUUID } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { sql } from "@vercel/postgres";

import {
  type CreateJobInput,
  type DatasetInfo,
  type JobSummary,
  CreateJobInputSchema,
  JobSummarySchema,
} from "@/lib/contracts/jobs";

const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY_MS = 100;
const MAX_RETRY_DELAY_MS = 2000;

interface RetryOptions {
  maxRetries?: number;
  initialDelayMs?: number;
  maxDelayMs?: number;
}

async function withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions = {}
): Promise<T> {
  const maxRetries = options.maxRetries ?? MAX_RETRIES;
  const initialDelayMs = options.initialDelayMs ?? INITIAL_RETRY_DELAY_MS;
  const maxDelayMs = options.maxDelayMs ?? MAX_RETRY_DELAY_MS;

  let attempt = 0;
  let lastError: unknown;

  while (attempt <= maxRetries) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries) {
        break;
      }

      const exponentialDelay = Math.min(initialDelayMs * 2 ** attempt, maxDelayMs);
      const jitter = Math.floor(
        Math.random() * Math.max(1, Math.floor(exponentialDelay * 0.2))
      );
      const delayMs = exponentialDelay + jitter;

      await new Promise((resolve) => setTimeout(resolve, delayMs));
      attempt += 1;
    }
  }

  throw lastError;
}

const datasetCatalog: DatasetInfo[] = [
  { id: "iris", label: "Iris", nClasses: 3, defaultQubits: 4 },
  {
    id: "breast_cancer",
    label: "Breast Cancer Wisconsin",
    nClasses: 2,
    defaultQubits: 8,
  },
  { id: "wine", label: "Wine", nClasses: 3, defaultQubits: 8 },
  { id: "mnist_binary", label: "MNIST Binary", nClasses: 2, defaultQubits: 16 },
  { id: "csv_upload", label: "CSV Upload", nClasses: 2, defaultQubits: 8 },
];

const workerBaseUrl = process.env.WORKER_BASE_URL ?? "http://localhost:8000";
const postgresEnabled = Boolean(process.env.POSTGRES_URL);
const qstashUrl = process.env.QSTASH_URL;
const qstashToken = process.env.QSTASH_TOKEN;
const appBaseUrl =
  process.env.APP_BASE_URL ??
  process.env.NEXT_PUBLIC_APP_URL ??
  "http://127.0.0.1:3000";
let schemaInitialized = false;
const localDataDir = path.join(process.cwd(), ".local-data");
const localJobsPath = path.join(localDataDir, "jobs.json");

function isPlaceholderWorkerUrl(url: string): boolean {
  if (!url) {
    return true;
  }

  if (/your-worker-domain|your-deployed-worker-url|example\.com/i.test(url)) {
    return true;
  }

  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();
    return (
      host === "your-worker-domain" ||
      host === "your-deployed-worker-url.com" ||
      host === "example.com" ||
      host.endsWith(".example.com")
    );
  } catch {
    return true;
  }
}

function nowIso(): string {
  return new Date().toISOString();
}

async function ensureSchema(): Promise<void> {
  if (!postgresEnabled || schemaInitialized) {
    return;
  }

  await sql`
    CREATE TABLE IF NOT EXISTS jobs (
      id TEXT PRIMARY KEY,
      job_type TEXT NOT NULL,
      dataset_id TEXT NOT NULL,
      status TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      message TEXT,
      result_json TEXT
    )
  `;

  schemaInitialized = true;
}

async function loadLocalJobs(): Promise<JobSummary[]> {
  try {
    const raw = await readFile(localJobsPath, "utf8");
    const parsed = JSON.parse(raw) as JobSummary[];
    return parsed.map((item) => JobSummarySchema.parse(item));
  } catch {
    return [];
  }
}

async function saveLocalJobs(jobs: JobSummary[]): Promise<void> {
  await mkdir(localDataDir, { recursive: true });
  await writeFile(localJobsPath, JSON.stringify(jobs, null, 2), "utf8");
}

function validate(input: unknown): CreateJobInput {
  return CreateJobInputSchema.parse(input);
}

function withTimestamps(
  id: string,
  input: CreateJobInput,
  status: JobSummary["status"],
  message?: string,
  result?: Record<string, unknown>
): JobSummary {
  return JobSummarySchema.parse({
    id,
    jobType: input.jobType,
    datasetId: input.datasetId,
    status,
    createdAt: nowIso(),
    updatedAt: nowIso(),
    message,
    result,
  });
}

function rowToJob(row: {
  id: string;
  job_type: string;
  dataset_id: string;
  status: string;
  created_at: Date | string;
  updated_at: Date | string;
  message: string | null;
  result_json: string | null;
}): JobSummary {
  return JobSummarySchema.parse({
    id: row.id,
    jobType: row.job_type,
    datasetId: row.dataset_id,
    status: row.status,
    createdAt: new Date(row.created_at).toISOString(),
    updatedAt: new Date(row.updated_at).toISOString(),
    message: row.message ?? undefined,
    result: row.result_json
      ? (JSON.parse(row.result_json) as Record<string, unknown>)
      : undefined,
  });
}

async function persist(job: JobSummary): Promise<void> {
  if (!postgresEnabled) {
    const current = await loadLocalJobs();
    const next = current.filter((item) => item.id !== job.id);
    next.push(job);
    await saveLocalJobs(next);
    return;
  }

  await ensureSchema();

  await sql`
    INSERT INTO jobs (id, job_type, dataset_id, status, created_at, updated_at, message, result_json)
    VALUES (
      ${job.id},
      ${job.jobType},
      ${job.datasetId},
      ${job.status},
      ${job.createdAt},
      ${job.updatedAt},
      ${job.message ?? null},
      ${job.result ? JSON.stringify(job.result) : null}
    )
    ON CONFLICT (id)
    DO UPDATE SET
      status = EXCLUDED.status,
      updated_at = EXCLUDED.updated_at,
      message = EXCLUDED.message,
      result_json = EXCLUDED.result_json
  `;
}

async function updateJob(id: string, updates: Partial<JobSummary>): Promise<void> {
  const existing = await getJob(id);
  if (!existing) {
    return;
  }

  await persist(
    JobSummarySchema.parse({
      ...existing,
      ...updates,
      updatedAt: nowIso(),
    })
  );
}

async function dispatchJob(id: string, input: CreateJobInput): Promise<void> {
  const existing = await getJob(id);
  if (!existing || existing.status === "completed") {
    return;
  }

  if (process.env.NODE_ENV === "production") {
    if (/localhost|127\.0\.0\.1/.test(workerBaseUrl)) {
      await updateJob(id, {
        status: "failed",
        message:
          "WORKER_BASE_URL points to localhost in production. Set it to your deployed worker URL.",
      });
      return;
    }

    if (isPlaceholderWorkerUrl(workerBaseUrl)) {
      await updateJob(id, {
        status: "failed",
        message:
          "WORKER_BASE_URL is not configured with a real public worker URL. Update Vercel env and redeploy.",
      });
      return;
    }
  }

  await updateJob(id, {
    status: "running",
    message: "Worker accepted the job.",
  });

  try {
    const response = await withRetry(async () => {
      const res = await fetch(`${workerBaseUrl}/jobs/run`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(input),
        cache: "no-store",
      });

      if (!res.ok && res.status >= 500) {
        throw new Error(`Worker temporary failure (${res.status})`);
      }

      return res;
    });

    if (!response.ok) {
      const body = await response.text();
      await updateJob(id, {
        status: "failed",
        message: `Worker request failed (${response.status}): ${body.slice(0, 180)}`,
      });
      return;
    }

    const payload = (await response.json()) as {
      status: string;
      result: Record<string, unknown>;
    };

    await updateJob(id, {
      status: "completed",
      message: "Completed by worker service.",
      result: payload.result,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    await updateJob(id, {
      status: "failed",
      message: `Worker dispatch error: ${message}`,
    });
  }
}

async function enqueueJob(id: string, input: CreateJobInput): Promise<void> {
  if (!qstashUrl || !qstashToken) {
    void dispatchJob(id, input);
    return;
  }

  const targetUrl = `${appBaseUrl}/api/internal/jobs/dispatch`;
  const publishUrl = `${qstashUrl.replace(/\/$/, "")}/v2/publish/${encodeURIComponent(targetUrl)}`;
  const dispatchSecret = process.env.WORKER_WEBHOOK_SECRET;
  const headers: Record<string, string> = {
    Authorization: `Bearer ${qstashToken}`,
    "Content-Type": "application/json",
    "Upstash-Retries": "3",
    "Upstash-Delay": "1s",
  };

  if (dispatchSecret) {
    headers["Upstash-Forward-x-dispatch-secret"] = dispatchSecret;
  }

  const response = await withRetry(async () => {
    const res = await fetch(publishUrl, {
      method: "POST",
      headers,
      body: JSON.stringify({
        id,
        input,
      }),
      cache: "no-store",
    });

    if (!res.ok && res.status >= 500) {
      throw new Error(`QStash temporary failure (${res.status})`);
    }

    return res;
  });

  if (!response.ok) {
    const body = await response.text();
    await updateJob(id, {
      status: "failed",
      message: `Queue publish failed (${response.status}): ${body.slice(0, 180)}`,
    });
    return;
  }

  await updateJob(id, {
    message: "Job queued via QStash.",
  });
}

export function listDatasets(): DatasetInfo[] {
  return datasetCatalog;
}

export async function listJobs(): Promise<JobSummary[]> {
  if (!postgresEnabled) {
    const current = await loadLocalJobs();
    return current.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  }

  await ensureSchema();
  const result = await sql<{
    id: string;
    job_type: string;
    dataset_id: string;
    status: string;
    created_at: Date | string;
    updated_at: Date | string;
    message: string | null;
    result_json: string | null;
  }>`
    SELECT id, job_type, dataset_id, status, created_at, updated_at, message, result_json
    FROM jobs
    ORDER BY created_at DESC
  `;

  return result.rows.map(rowToJob);
}

export async function getJob(id: string): Promise<JobSummary | null> {
  if (!postgresEnabled) {
    const current = await loadLocalJobs();
    return current.find((item) => item.id === id) ?? null;
  }

  await ensureSchema();
  const result = await sql<{
    id: string;
    job_type: string;
    dataset_id: string;
    status: string;
    created_at: Date | string;
    updated_at: Date | string;
    message: string | null;
    result_json: string | null;
  }>`
    SELECT id, job_type, dataset_id, status, created_at, updated_at, message, result_json
    FROM jobs
    WHERE id = ${id}
    LIMIT 1
  `;

  if (result.rows.length === 0) {
    return null;
  }

  return rowToJob(result.rows[0]);
}

export async function createJob(rawInput: unknown): Promise<JobSummary> {
  const input = validate(rawInput);
  const id = randomUUID();
  const queued = withTimestamps(id, input, "queued", "Job is waiting in queue.");

  await persist(queued);
  void enqueueJob(id, input);

  return queued;
}

export async function executeQueuedJob(id: string, rawInput: unknown): Promise<void> {
  const input = validate(rawInput);
  await dispatchJob(id, input);
}
