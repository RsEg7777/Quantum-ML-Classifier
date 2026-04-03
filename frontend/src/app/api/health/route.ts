import { NextResponse } from "next/server";

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

export async function GET() {
  const workerBaseUrl = process.env.WORKER_BASE_URL ?? "http://localhost:8000";
  const workerHealthUrl = `${workerBaseUrl.replace(/\/$/, "")}/health`;
  const usingLocalWorkerInProd =
    process.env.NODE_ENV === "production" &&
    /localhost|127\.0\.0\.1/.test(workerBaseUrl);
  const workerBaseUrlPlaceholder = isPlaceholderWorkerUrl(workerBaseUrl);

  let worker = {
    reachable: false,
    status: "unreachable",
    url: workerHealthUrl,
    details: null as Record<string, unknown> | null,
  };

  if (workerBaseUrlPlaceholder) {
    worker = {
      reachable: false,
      status: "misconfigured_placeholder_url",
      url: workerHealthUrl,
      details: null,
    };
  } else {
    try {
      const response = await fetch(workerHealthUrl, {
        method: "GET",
        cache: "no-store",
        signal: AbortSignal.timeout(2500),
      });

      if (response.ok) {
        const payload = (await response.json()) as Record<string, unknown>;
        worker = {
          reachable: true,
          status: "healthy",
          url: workerHealthUrl,
          details: payload,
        };
      } else {
        worker = {
          reachable: false,
          status: `error_${response.status}`,
          url: workerHealthUrl,
          details: null,
        };
      }
    } catch {
      worker = {
        reachable: false,
        status: "timeout_or_network_error",
        url: workerHealthUrl,
        details: null,
      };
    }
  }

  return NextResponse.json({
    status: "ok",
    service: "frontend-api",
    timestamp: new Date().toISOString(),
    worker,
    config: {
      hasAuthSecret: Boolean(process.env.AUTH_SECRET),
      hasWorkerWebhookSecret: Boolean(process.env.WORKER_WEBHOOK_SECRET),
      hasBlobToken: Boolean(process.env.BLOB_READ_WRITE_TOKEN),
      hasPostgresUrl: Boolean(process.env.POSTGRES_URL),
      usingLocalWorkerInProd,
      workerBaseUrlPlaceholder,
    },
  });
}
