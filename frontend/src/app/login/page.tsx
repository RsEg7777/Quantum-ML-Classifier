"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
    <main className="page-shell flex min-h-screen items-center justify-center px-6 py-16">
      <section className="card-glass w-full max-w-md p-8">
        <h1 className="text-3xl font-semibold text-slate-900">Sign In</h1>
        <p className="mt-2 text-sm text-slate-600">
          Access the Quantum ML operations dashboard.
        </p>

        <form className="mt-6 space-y-4" onSubmit={onSubmit}>
          <label className="block text-sm font-medium text-slate-800">
            Email
            <input
              type="email"
              required
              className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </label>

          <label className="block text-sm font-medium text-slate-800">
            Password
            <input
              type="password"
              required
              className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </label>

          <button className="btn-primary w-full" type="submit" disabled={isLoading}>
            {isLoading ? "Signing in..." : "Sign in"}
          </button>
        </form>

        {error ? (
          <p className="mt-4 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </p>
        ) : null}
      </section>
    </main>
  );
}
