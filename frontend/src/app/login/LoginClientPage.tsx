"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

type LoginClientPageProps = {
  oauthErrorCode: string | null;
};

function getOAuthErrorMessage(errorCode: string | null): string | null {
  if (!errorCode) {
    return null;
  }

  switch (errorCode) {
    case "google_not_configured":
      return "Google sign-in is not configured yet. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.";
    case "google_access_denied":
      return "Google sign-in was canceled. Please try again.";
    case "google_state_mismatch":
      return "Google sign-in state validation failed. Please retry from this page.";
    case "google_token_exchange_failed":
    case "google_token_missing":
    case "google_profile_failed":
      return "Unable to complete Google sign-in right now. Please try again.";
    case "google_email_unverified":
      return "Your Google account email is not verified.";
    case "google_email_not_allowed":
      return "This Google account is not allowed for this workspace.";
    default:
      return "Google sign-in failed. Please try again.";
  }
}

export default function LoginClientPage({ oauthErrorCode }: LoginClientPageProps) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const oauthError = getOAuthErrorMessage(oauthErrorCode);

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsLoading(true);
    setError(null);

    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      setError("Invalid credentials. Please try again.");
      setIsLoading(false);
      return;
    }

    router.push("/dashboard");
    router.refresh();
  }

  return (
    <main className="page-shell flex min-h-screen items-center justify-center px-6 py-16 md:px-10">
      <span className="ambient-orb orb-a" aria-hidden="true" />
      <span className="ambient-orb orb-b" aria-hidden="true" />

      <section className="card-glass fade-up w-full max-w-md p-8 md:p-9">
        <p className="kicker-badge">SECURE ACCESS</p>
        <h1 className="mt-4 text-3xl font-semibold text-slate-900">Sign in</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-700">
          Authenticate to manage quantum-classical jobs and monitor queue state in
          real time.
        </p>

        <div className="mt-6 glow-separator" />

        <a className="btn-google mt-6 w-full" href="/api/auth/google/login">
          <svg
            className="btn-google-icon"
            viewBox="0 0 24 24"
            aria-hidden="true"
            focusable="false"
          >
            <path
              fill="#EA4335"
              d="M12 10.2v3.9h5.5c-.2 1.2-1.4 3.6-5.5 3.6-3.3 0-6-2.7-6-6.1s2.7-6.1 6-6.1c1.9 0 3.2.8 3.9 1.5l2.7-2.6C16.9 2.9 14.7 2 12 2 6.9 2 2.8 6.2 2.8 11.4S6.9 20.8 12 20.8c6.9 0 9.1-4.8 9.1-7.3 0-.5-.1-.9-.1-1.3z"
            />
            <path
              fill="#34A853"
              d="M3.7 7.3l3.2 2.3c.9-1.8 2.8-3 5.1-3 1.9 0 3.2.8 3.9 1.5l2.7-2.6C16.9 2.9 14.7 2 12 2 8 2 4.6 4.3 3 7.7z"
            />
            <path
              fill="#4A90E2"
              d="M12 20.8c2.6 0 4.8-.8 6.4-2.3L15.4 16c-.9.6-2.1 1.1-3.4 1.1-2.9 0-5.2-1.9-6-4.5L2.8 15c1.6 3.3 5 5.8 9.2 5.8z"
            />
            <path
              fill="#FBBC05"
              d="M6 12.6c-.2-.5-.3-1-.3-1.5 0-.5.1-1.1.3-1.5L2.8 7.3C2.2 8.5 1.8 9.9 1.8 11.4s.4 2.9 1.1 4.1z"
            />
          </svg>
          Continue with Google
        </a>

        <div className="my-5 flex items-center gap-3">
          <span className="h-px flex-1 bg-slate-300/70" />
          <span className="text-xs uppercase tracking-[0.12em] text-slate-500">
            Or continue with email
          </span>
          <span className="h-px flex-1 bg-slate-300/70" />
        </div>

        <form className="space-y-4" onSubmit={onSubmit}>
          <label className="field-label">
            Email
            <input
              type="email"
              required
              autoComplete="email"
              className="input-glass"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>

          <label className="field-label">
            Password
            <input
              type="password"
              required
              autoComplete="current-password"
              className="input-glass"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>

          <button className="btn-primary w-full" type="submit" disabled={isLoading}>
            {isLoading ? "Signing in..." : "Sign in"}
          </button>
        </form>

        {oauthError ? <p className="alert-error text-sm">{oauthError}</p> : null}
        {error ? <p className="alert-error text-sm">{error}</p> : null}
      </section>
    </main>
  );
}
