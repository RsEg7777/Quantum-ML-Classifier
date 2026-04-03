"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import type { DatasetInfo, JobSummary, JobType } from "@/lib/contracts/jobs";

type JobsResponse = { jobs: JobSummary[] };
type DatasetsResponse = { datasets: DatasetInfo[] };
type HealthResponse = {
  worker?: {
    reachable?: boolean;
    status?: string;
    url?: string;
  };
  config?: {
    usingLocalWorkerInProd?: boolean;
  };
};

type FormState = {
  jobType: JobType;
  datasetId: DatasetInfo["id"];
  configJson: string;
};

export default function DashboardPage() {
  const router = useRouter();
  const initialized = useRef(false);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [csvBlobUrl, setCsvBlobUrl] = useState<string | null>(null);
  const [csvFilename, setCsvFilename] = useState<string | null>(null);
  const [workerReachable, setWorkerReachable] = useState<boolean | null>(null);
  const [workerStatus, setWorkerStatus] = useState<string>("unknown");
  const [workerEndpoint, setWorkerEndpoint] = useState<string>("-");
  const [workerMisconfigured, setWorkerMisconfigured] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState<FormState>({
    jobType: "training",
    datasetId: "iris",
    configJson: '{"n_qubits": 4, "n_layers": 3}',
  });

  const metrics = useMemo(() => {
    const total = jobs.length;
    const running = jobs.filter((job) => job.status === "running").length;
    const completed = jobs.filter((job) => job.status === "completed").length;
    const failed = jobs.filter((job) => job.status === "failed").length;

    return { total, running, completed, failed };
  }, [jobs]);

  async function loadDatasets() {
    const response = await fetch("/api/datasets", { cache: "no-store" });
    if (!response.ok) {
      setError("Failed to load dataset catalog.");
      return;
    }

    const payload = (await response.json()) as DatasetsResponse;
    setDatasets(payload.datasets);
  }

  async function loadJobs() {
    const response = await fetch("/api/jobs", { cache: "no-store" });
    if (!response.ok) {
      setError("Failed to load jobs.");
      return;
    }

    const payload = (await response.json()) as JobsResponse;
    setJobs(payload.jobs);
  }

  async function loadHealth() {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) {
      setWorkerReachable(false);
      setWorkerStatus(`health_error_${response.status}`);
      return;
    }

    const payload = (await response.json()) as HealthResponse;
    setWorkerReachable(Boolean(payload.worker?.reachable));
    setWorkerStatus(payload.worker?.status ?? "unknown");
    setWorkerEndpoint(payload.worker?.url ?? "-");
    setWorkerMisconfigured(Boolean(payload.config?.usingLocalWorkerInProd));
  }

  async function submitJob(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);

    let parsedConfig: Record<string, unknown>;

    try {
      parsedConfig = JSON.parse(form.configJson) as Record<string, unknown>;
    } catch {
      setIsSubmitting(false);
      setError("Config must be valid JSON.");
      return;
    }

    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jobType: form.jobType,
        datasetId: form.datasetId,
        config: parsedConfig,
        csvBlobUrl: form.datasetId === "csv_upload" ? csvBlobUrl : undefined,
      }),
    });

    if (!response.ok) {
      const data = (await response.json()) as { error?: string };
      setError(data.error ?? "Unable to submit job.");
      setIsSubmitting(false);
      return;
    }

    await loadJobs();
    setIsSubmitting(false);
  }

  async function uploadCsv(file: File) {
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/uploads", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const data = (await response.json()) as { error?: string };
      setError(data.error ?? "Failed to upload CSV.");
      setIsUploading(false);
      return;
    }

    const payload = (await response.json()) as {
      csvBlobUrl: string;
      filename: string;
    };

    setCsvBlobUrl(payload.csvBlobUrl);
    setCsvFilename(payload.filename);
    setIsUploading(false);
  }

  async function signOut() {
    await fetch("/api/auth/logout", {
      method: "POST",
    });
    router.push("/login");
    router.refresh();
  }

  useEffect(() => {
    // One-time initialization on mount
    if (!initialized.current) {
      initialized.current = true;
      const init = async () => {
        await loadDatasets();
        await loadJobs();
        await loadHealth();
      };
      init().catch((err) => setError("Failed to initialize: " + String(err)));
    }
  }, []);

  // Smart polling effect: adjust based on job states
  useEffect(() => {
    const hasRunningJobs = jobs.some((job) => job.status === "running");
    const pollInterval = hasRunningJobs ? 500 : 5000; // 500ms when running, 5s when idle
    
    const interval = setInterval(() => {
      void loadJobs();
    }, pollInterval);

    return () => clearInterval(interval);
  }, [jobs]);

  return (
    <main className="page-shell min-h-screen px-6 py-8 md:px-10">
      <span className="ambient-orb orb-a" aria-hidden="true" />
      <span className="ambient-orb orb-b" aria-hidden="true" />

      <section className="mx-auto grid max-w-7xl gap-6 lg:grid-cols-[1.08fr_1.72fr]">
        <article className="card-glass fade-up p-6 md:p-8">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="kicker-badge">JOB ORCHESTRATION</p>
              <h1 className="mt-3 text-3xl font-semibold text-slate-900">Job Builder</h1>
            </div>
            <button className="btn-ghost" type="button" onClick={signOut}>
              Sign out
            </button>
          </div>

          <p className="mt-3 text-sm leading-relaxed text-slate-700">
            Submit training, inference, or experiment workloads into the queue with
            dataset-aware configuration.
          </p>

          <div className="mt-5 glow-separator" />

          <form className="mt-5 space-y-4" onSubmit={submitJob}>
            <label className="field-label">
              Job Type
              <select
                className="select-glass"
                value={form.jobType}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    jobType: event.target.value as JobType,
                  }))
                }
              >
                <option value="training">Training</option>
                <option value="inference">Inference</option>
                <option value="experiment">Experiment</option>
              </select>
            </label>

            <label className="field-label">
              Dataset
              <select
                className="select-glass"
                value={form.datasetId}
                onChange={(event) => {
                  const nextDataset = event.target.value as DatasetInfo["id"];
                  if (nextDataset !== "csv_upload") {
                    setCsvBlobUrl(null);
                    setCsvFilename(null);
                  }
                  setForm((current) => ({
                    ...current,
                    datasetId: nextDataset,
                  }));
                }}
              >
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.label}
                  </option>
                ))}
              </select>
            </label>

            {form.datasetId === "csv_upload" ? (
              <label className="field-label">
                CSV File
                <input
                  type="file"
                  accept=".csv,text/csv"
                  className="input-glass"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) {
                      void uploadCsv(file);
                    }
                  }}
                />
                <div className="surface-muted mt-2 px-3 py-2">
                  <p className="text-xs text-slate-600">
                    Expected format: header row, numeric features, and label in last column.
                  </p>
                  {isUploading ? (
                    <p className="mt-1 text-xs text-slate-700">Uploading...</p>
                  ) : null}
                  {csvFilename ? (
                    <p className="mt-1 text-xs font-medium text-emerald-700">Uploaded: {csvFilename}</p>
                  ) : null}
                </div>
              </label>
            ) : null}

            <label className="field-label">
              Config JSON
              <textarea
                className="textarea-glass font-mono text-xs"
                value={form.configJson}
                onChange={(event) =>
                  setForm((current) => ({
                    ...current,
                    configJson: event.target.value,
                  }))
                }
              />
            </label>

            <button
              className="btn-primary w-full"
              type="submit"
              disabled={
                isSubmitting ||
                isUploading ||
                (form.datasetId === "csv_upload" && !csvBlobUrl)
              }
            >
              {isSubmitting ? "Submitting..." : "Submit Job"}
            </button>
          </form>

          {error ? <p className="alert-error text-sm">{error}</p> : null}
        </article>

        <article className="card-glass fade-up delay-1 p-6 md:p-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-2xl font-semibold text-slate-900">Live Queue</h2>
              <p className="mt-1 text-xs uppercase tracking-[0.12em] text-slate-500">
                Polling automatically while jobs are running
              </p>
            </div>
            <button className="btn-ghost" type="button" onClick={() => void loadJobs()}>
              Refresh
            </button>
          </div>

          <div className="mt-4 grid gap-3 sm:grid-cols-4">
            <Metric label="Total" value={metrics.total} />
            <Metric label="Running" value={metrics.running} />
            <Metric label="Completed" value={metrics.completed} />
            <Metric label="Failed" value={metrics.failed} />
          </div>

          <div className="surface-muted mt-4 flex flex-wrap items-center justify-between gap-3 px-4 py-3">
            <div className="flex items-center gap-2">
              <span
                className={`health-dot ${workerReachable ? "online" : "offline"}`}
                aria-hidden="true"
              />
              <p className="text-sm text-slate-700">
                Worker connectivity: <strong>{workerReachable ? "Online" : "Offline"}</strong>
              </p>
            </div>
            <p className="text-xs text-slate-600">{workerStatus}</p>
          </div>

          <p className="mt-2 text-xs text-slate-500">Endpoint: {workerEndpoint}</p>
          {workerMisconfigured ? (
            <p className="mt-1 text-xs font-semibold text-amber-700">
              Warning: production is pointing to localhost worker URL.
            </p>
          ) : null}

          <div className="mt-6 space-y-3">
            {jobs.length === 0 ? (
              <p className="surface-muted px-4 py-6 text-center text-sm text-slate-700">
                No jobs yet. Submit one from the panel on the left.
              </p>
            ) : (
              jobs.map((job, index) => (
                <article
                  key={job.id}
                  className={`queue-item fade-up ${index < 3 ? `delay-${index + 1}` : ""}`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <h3 className="font-medium text-slate-900">{job.id.slice(0, 8)}</h3>
                    <span className={`status-pill ${job.status}`}>{job.status}</span>
                  </div>

                  <p className="mt-2 text-sm text-slate-700">
                    {job.jobType} on {job.datasetId}
                  </p>

                  <p className="mt-1 text-xs text-slate-600">{job.message ?? "No message"}</p>

                  {job.result ? (
                    <pre className="result-panel text-xs">{JSON.stringify(job.result, null, 2)}</pre>
                  ) : null}
                </article>
              ))
            )}
          </div>
        </article>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="metric-card">
      <p className="text-xs uppercase tracking-[0.12em] text-slate-500">{label}</p>
      <p className="text-2xl font-semibold text-slate-900">{value}</p>
    </div>
  );
}
